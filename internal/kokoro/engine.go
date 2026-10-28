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
	// durationsPath is the side-car graph that emits the duration predictor tensor,
	// or "" when it is not installed. Without it word timings fall back to silence
	// analysis of the PCM.
	durationsPath string
}

// DurationsModelPath is the conventional location of the Kokoro duration side-car next
// to an audio model: <model>-durations.onnx.
func DurationsModelPath(modelPath string) string {
	return strings.TrimSuffix(modelPath, ".onnx") + "-durations.onnx"
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

	if p := DurationsModelPath(modelPath); fileExists(p) {
		e.durationsPath = p
	}
	return nil
}

func fileExists(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir() && fi.Size() > 200
}

// synthesize runs the audio graph once and, when the duration side-car is installed,
// the duration graph with the very same tokens, style row and speed. Returning both
// from one call is what keeps ?timestamps honest: one request, one synthesis, and
// timings that cannot belong to a different pass than the audio.
func (e *KokoroEngine) synthesize(text string, voice string, speed float32) ([]float32, []int64, *shared.TokenStream, error) {
	matchedVoice := findVoiceMatch(voice)
	if matchedVoice == "" {
		matchedVoice = strings.ToLower(voice)
	}

	ts := shared.BuildTokenStream(text, matchedVoice)
	tokens := ts.IDs

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
		return nil, nil, ts, fmt.Errorf("no tokens mapped from text")
	}

	tokensShape := ort.NewShape(1, int64(len(tokens)))
	tokensTensor, err := ort.NewTensor(tokensShape, tokens)
	if err != nil {
		return nil, nil, ts, err
	}
	defer tokensTensor.Destroy()

	styleShape := ort.NewShape(1, 256)
	styleTensor, err := ort.NewTensor(styleShape, styleVec)
	if err != nil {
		return nil, nil, ts, err
	}
	defer styleTensor.Destroy()

	speedShape := ort.NewShape(1)
	speedTensor, err := ort.NewTensor(speedShape, []float32{speed})
	if err != nil {
		return nil, nil, ts, err
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
		return nil, nil, ts, fmt.Errorf("failed to create dynamic session: %v", err)
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
		return nil, nil, ts, fmt.Errorf("failed to run dynamic session: %v", err)
	}

	if outputs[0] == nil {
		return nil, nil, ts, fmt.Errorf("dynamic session did not allocate output tensor")
	}
	defer outputs[0].Destroy()

	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, ts, fmt.Errorf("failed to cast output tensor to float32")
	}

	outputData := outTensor.GetData()

	// The duration side-graph takes exactly the same inputs as the audio graph, so the
	// frames it reports describe the waveform produced above. Speed is declared as a
	// scalar input; a [1] tensor is accepted for it.
	var durations []int64
	if e.durationsPath != "" {
		if d, derr := e.runDurations(inputs); derr == nil {
			durations = d
		}
	}

	return outputData, durations, ts, nil
}

// runDurations executes the duration side-graph with the tokens/style/speed already
// prepared for the audio call. Cost is ~4% of the audio graph.
func (e *KokoroEngine) runDurations(audioInputs []ort.Value) ([]int64, error) {
	session, err := ort.NewDynamicAdvancedSession(e.durationsPath,
		[]string{"tokens", "style", "speed"}, []string{"durations"}, nil)
	if err != nil {
		return nil, fmt.Errorf("durations session: %v", err)
	}
	defer session.Destroy()

	outputs := []ort.Value{nil}
	if err := session.Run(audioInputs, outputs); err != nil {
		return nil, fmt.Errorf("durations run: %v", err)
	}
	if outputs[0] == nil {
		return nil, fmt.Errorf("durations session did not allocate output tensor")
	}
	defer outputs[0].Destroy()

	t, ok := outputs[0].(*ort.Tensor[int64])
	if !ok {
		return nil, fmt.Errorf("failed to cast durations tensor to int64")
	}
	return t.GetData(), nil
}

func (e *KokoroEngine) Synthesize(text string, voice string, speed float32) ([]float32, int, error) {
	outputData, _, _, err := e.synthesize(text, voice, speed)
	if err != nil {
		return nil, 0, err
	}

	// Clean up trailing silent zeros
	outputData = trimTrailingSilence(outputData)
	return outputData, e.sampleRate, nil
}

// trimTrailingSilence drops the exact-zero samples the graph pads the end with.
func trimTrailingSilence(data []float32) []float32 {
	var lastNonZero int = len(data) - 1
	for lastNonZero >= 0 && data[lastNonZero] == 0 {
		lastNonZero--
	}
	if lastNonZero >= 0 {
		return data[:lastNonZero+1]
	}
	return data
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

// SynthesizeWithTimings implements TimingEngine. It returns the audio of a single
// synthesis pass together with per-word timestamps: exact ones when the duration
// side-car is installed, silence-derived estimates otherwise.
func (e *KokoroEngine) SynthesizeWithTimings(text, voice string, speed float32) ([]float32, []shared.WordTiming, int, error) {
	alignment, err := e.SynthesizeAligned(text, voice, speed)
	if err != nil {
		return nil, nil, 0, err
	}
	return alignment.Samples, alignment.Words, alignment.SampleRate, nil
}

// SynthesizeAligned synthesizes once and returns word (and phoneme) timings taken from
// the duration predictor of the model itself, falling back to silence analysis when the
// kokoro-v1.0-durations.onnx side-car is not installed.
func (e *KokoroEngine) SynthesizeAligned(text, voice string, speed float32) (shared.Alignment, error) {
	raw, durations, ts, err := e.synthesize(text, voice, speed)
	if err != nil {
		return shared.Alignment{}, err
	}
	matchedVoice := findVoiceMatch(voice)
	if matchedVoice == "" {
		matchedVoice = strings.ToLower(voice)
	}

	samples := trimTrailingSilence(raw)
	alignment := shared.Alignment{
		Samples:    samples,
		SampleRate: e.sampleRate,
		Source:     shared.TimingSourceVAD,
		// Overwritten with the model frame size as soon as the duration tensor is used.
		FrameMS: float64(shared.VADWindowMS),
	}

	if len(durations) > 0 {
		words, phonemes, frameMS, aerr := shared.TimingsFromDurations(durations, ts, e.sampleRate, len(raw), len(samples))
		if aerr == nil {
			alignment.Words = words
			alignment.Phonemes = phonemes
			alignment.Source = shared.TimingSourceDuration
			alignment.FrameMS = frameMS
			return alignment, nil
		}
		fmt.Printf("[Kokoro] Duration alignment unavailable (%v); using silence analysis.\n", aerr)
	}

	boundaries, ok := ts.WordBoundaries()
	if !ok {
		// The space split did not line up with the source words; per-word phonemization
		// is the only way left to get a token-count prior.
		boundaries = shared.TokenizeWordBoundaries(text, matchedVoice)
	}
	if len(boundaries) == 0 {
		return alignment, nil
	}
	alignment.Words = shared.SilenceBoundaryTimings(samples, e.sampleRate, boundaries)
	return alignment, nil
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
