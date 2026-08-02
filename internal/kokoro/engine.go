package kokoro

import (
	"archive/zip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"zen-tts/internal/shared"

	ort "github.com/yalue/onnxruntime_go"
)

type KokoroEngine struct {
	modelPath  string
	voices     map[string][]float32
	sampleRate int
}

func (e *KokoroEngine) Initialize(modelPath string, configPath string) error {
	if err := shared.InitONNXRuntime(); err != nil {
		return fmt.Errorf("failed to init onnxruntime: %v", err)
	}

	e.modelPath = modelPath
	e.sampleRate = 24000
	e.voices = make(map[string][]float32)

	// Load voice style vectors
	voicesDir := filepath.Join(filepath.Dir(modelPath), "voices")
	files, err := os.ReadDir(voicesDir)
	if err == nil {
		for _, f := range files {
			if !f.IsDir() && filepath.Ext(f.Name()) == ".bin" {
				vName := strings.TrimSuffix(f.Name(), filepath.Ext(f.Name()))
				vPath := filepath.Join(voicesDir, f.Name())
				vec, err := loadVoiceVector(vPath)
				if err != nil {
					fmt.Printf("[Kokoro] Error loading voice style '%s': %v. Deleting corrupt file.\n", f.Name(), err)
					os.Remove(vPath)
				} else {
					e.voices[strings.ToLower(vName)] = vec
				}
			}
		}
	}
	return nil
}

func (e *KokoroEngine) Synthesize(text string, voice string, speed float32) ([]float32, int, error) {
	matchedVoice := findVoiceMatch(voice)
	if matchedVoice == "" {
		matchedVoice = strings.ToLower(voice)
	}

	tokens := shared.TokenizeWithVoice(text, matchedVoice)

	voiceData, ok := e.voices[matchedVoice]
	if !ok || len(voiceData) == 0 {
		voicesDir := filepath.Join(filepath.Dir(e.modelPath), "voices")
		binPath := filepath.Join(voicesDir, matchedVoice+".bin")

		if _, err := os.Stat(binPath); os.IsNotExist(err) {
			if KokoroVoices[matchedVoice] {
				fmt.Printf("[Kokoro] Voice '%s' not found locally. Fetching on-demand...\n", matchedVoice)
				if err := downloadAndExtractVoice(voicesDir, matchedVoice); err != nil {
					fmt.Printf("[Kokoro] Error fetching voice '%s': %v\n", matchedVoice, err)
				}
			}
		}

		if _, err := os.Stat(binPath); err == nil {
			vec, err := loadVoiceVector(binPath)
			if err == nil {
				e.voices[matchedVoice] = vec
				voiceData = vec
			}
		}

		if len(voiceData) == 0 {
			for k, vec := range e.voices {
				if strings.Contains(k, matchedVoice) || strings.Contains(matchedVoice, k) {
					voiceData = vec
					break
				}
			}
			if len(voiceData) == 0 {
				for _, vec := range e.voices {
					voiceData = vec
					break
				}
			}
		}
	}

	// Extract voice style embedding corresponding to the token sequence length
	styleVec := make([]float32, 256)
	numEmbeds := len(voiceData) / 256
	if numEmbeds > 0 {
		idx := len(tokens)
		if idx >= numEmbeds {
			idx = numEmbeds - 1
		}
		copy(styleVec, voiceData[idx*256:(idx+1)*256])
	} else if len(voiceData) >= 256 {
		copy(styleVec, voiceData[:256])
	}

	if len(tokens) == 0 {
		return nil, e.sampleRate, fmt.Errorf("no tokens mapped from text")
	}

	tokensShape := ort.NewShape(1, int64(len(tokens)))
	tokensTensor, err := ort.NewTensor(tokensShape, tokens)
	if err != nil {
		return nil, 0, err
	}
	defer tokensTensor.Destroy()

	styleShape := ort.NewShape(1, 256)
	styleTensor, err := ort.NewTensor(styleShape, styleVec)
	if err != nil {
		return nil, 0, err
	}
	defer styleTensor.Destroy()

	speedShape := ort.NewShape(1)
	speedTensor, err := ort.NewTensor(speedShape, []float32{speed})
	if err != nil {
		return nil, 0, err
	}
	defer speedTensor.Destroy()

	var hasSpeed bool = true
	// Initialize dynamic advanced session
	session, err := ort.NewDynamicAdvancedSession(e.modelPath,
		[]string{"tokens", "style", "speed"},
		[]string{"audio"},
		nil)
	if err != nil {
		hasSpeed = false
		// Fallback to "tokens", "style" only
		session, err = ort.NewDynamicAdvancedSession(e.modelPath,
			[]string{"tokens", "style"},
			[]string{"audio"},
			nil)
	}

	if err != nil {
		return nil, 0, fmt.Errorf("failed to create dynamic session: %v", err)
	}
	defer session.Destroy()

	// Let the ONNX Runtime auto-allocate the dynamic outputs!
	var inputs []ort.Value
	if hasSpeed {
		inputs = []ort.Value{tokensTensor, styleTensor, speedTensor}
	} else {
		inputs = []ort.Value{tokensTensor, styleTensor}
	}
	outputs := []ort.Value{nil}

	err = session.Run(inputs, outputs)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to run dynamic session: %v", err)
	}

	if outputs[0] == nil {
		return nil, 0, fmt.Errorf("dynamic session did not allocate output tensor")
	}
	defer outputs[0].Destroy()

	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, fmt.Errorf("failed to cast output tensor to float32")
	}

	outputData := outTensor.GetData()

	// Clean up trailing silent zeros
	var lastNonZero int = len(outputData) - 1
	for lastNonZero >= 0 && outputData[lastNonZero] == 0 {
		lastNonZero--
	}
	if lastNonZero >= 0 {
		outputData = outputData[:lastNonZero+1]
	}

	return outputData, e.sampleRate, nil
}

func (e *KokoroEngine) SampleRate() int {
	return e.sampleRate
}

func (e *KokoroEngine) Close() error {
	return nil
}

func (e *KokoroEngine) SynthesizeStream(text string, voice string, speed float32, callback func(samples []float32) bool) error {
	samples, _, err := e.Synthesize(text, voice, speed)
	if err != nil {
		return err
	}
	callback(samples)
	return nil
}

// SynthesizeWithTimings implements TimingEngine. It synthesizes audio and then
// derives per-word timestamps by analyzing silence boundaries in the PCM output.
// Timings are relative to segment start (start at 0.0).
func (e *KokoroEngine) SynthesizeWithTimings(text, voice string, speed float32) ([]float32, []shared.WordTiming, int, error) {
	samples, sampleRate, err := e.Synthesize(text, voice, speed)
	if err != nil {
		return nil, nil, 0, err
	}

	matchedVoice := findVoiceMatch(voice)
	if matchedVoice == "" {
		matchedVoice = strings.ToLower(voice)
	}

	boundaries := shared.TokenizeWordBoundaries(text, matchedVoice)
	if len(boundaries) == 0 {
		return samples, nil, sampleRate, nil
	}

	timings := shared.SilenceBoundaryTimings(samples, sampleRate, boundaries)
	return samples, timings, sampleRate, nil
}

func loadVoiceVector(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var vec []float32
	if err := json.Unmarshal(data, &vec); err == nil && len(vec) >= 256 {
		return vec[:256], nil
	}
	if len(data) >= 1024 {
		count := len(data) / 4
		vec = make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
			vec[i] = math.Float32frombits(bits)
		}
		return vec, nil
	}
	return nil, fmt.Errorf("invalid voice style file size/format")
}

var KokoroVoices = map[string]bool{
	"af_alloy": true, "af_aoede": true, "af_bella": true, "af_heart": true,
	"af_jessica": true, "af_kore": true, "af_nicole": true, "af_nova": true,
	"af_river": true, "af_sarah": true, "af_sky": true, "am_adam": true,
	"am_echo": true, "am_eric": true, "am_fenrir": true, "am_liam": true,
	"am_michael": true, "am_onyx": true, "am_puck": true, "am_santa": true,
	"bf_alice": true, "bf_emma": true, "bf_isabella": true, "bf_lily": true,
	"bm_daniel": true, "bm_fable": true, "bm_george": true, "bm_lewis": true,
	"ef_dora": true, "em_alex": true, "em_santa": true, "ff_siwis": true,
	"hf_alpha": true, "hf_beta": true, "hm_omega": true, "hm_psi": true,
	"if_sara": true, "im_nicola": true, "jf_alpha": true, "jf_gongitsune": true,
	"jf_nezumi": true, "jf_tebukuro": true, "jm_kumo": true, "pf_dora": true,
	"pm_alex": true, "pm_santa": true, "zf_xiaobei": true, "zf_xiaoni": true,
	"zf_xiaoxiao": true, "zf_xiaoyi": true, "zm_yunjian": true, "zm_yunxi": true,
	"zm_yunxia": true, "zm_yunyang": true,
}

func findVoiceMatch(voice string) string {
	voiceLower := strings.ToLower(voice)
	if voiceLower == "" {
		return ""
	}
	if KokoroVoices[voiceLower] {
		return voiceLower
	}
	for kv := range KokoroVoices {
		if strings.Contains(kv, voiceLower) {
			return kv
		}
	}
	return ""
}

func downloadAndExtractVoice(voicesDir string, voiceName string) error {
	ptPath := filepath.Join(voicesDir, voiceName+".pt")
	binPath := filepath.Join(voicesDir, voiceName+".bin")

	if err := os.MkdirAll(voicesDir, 0755); err != nil {
		return err
	}

	url := fmt.Sprintf("https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/%s.pt", voiceName)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status code: %d", resp.StatusCode)
	}

	out, err := os.Create(ptPath)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, resp.Body)
	out.Close()
	if err != nil {
		os.Remove(ptPath)
		return err
	}

	r, err := zip.OpenReader(ptPath)
	if err != nil {
		os.Remove(ptPath)
		return err
	}
	defer r.Close()

	var dataFile *zip.File
	for _, f := range r.File {
		if strings.HasSuffix(f.Name, "/data/0") {
			dataFile = f
			break
		}
	}

	if dataFile == nil {
		os.Remove(ptPath)
		return fmt.Errorf("could not find tensor data in pt file")
	}

	rc, err := dataFile.Open()
	if err != nil {
		os.Remove(ptPath)
		return err
	}
	defer rc.Close()

	binOut, err := os.Create(binPath)
	if err != nil {
		os.Remove(ptPath)
		return err
	}
	_, err = io.Copy(binOut, rc)
	binOut.Close()

	os.Remove(ptPath)
	return nil
}
