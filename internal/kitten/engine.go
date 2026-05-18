package kitten

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func InitONNXRuntime() error {
	ortInitOnce.Do(func() {
		libPath := "./piper/libonnxruntime.so.1.24.2"
		if _, err := os.Stat(libPath); err != nil {
			libPath = "./piper/libonnxruntime.so"
		}
		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

type KittenEngine struct {
	modelPath  string
	vocab      map[string]int64
	voices     map[string][]float32
	sampleRate int
}

func (e *KittenEngine) Initialize(modelPath string, configPath string) error {
	if err := InitONNXRuntime(); err != nil {
		return fmt.Errorf("failed to init onnxruntime: %v", err)
	}

	e.modelPath = modelPath
	e.sampleRate = 24000 // KittenTTS is always 24 kHz
	e.voices = make(map[string][]float32)

	// Explicitly set the 100% correct, official KittenTTS/Kokoro IPA vocabulary
	e.vocab = map[string]int64{
		"$": 0, ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10,
		"\"": 11, "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "̃": 17, "ʣ": 18,
		"ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "ꭧ": 23, "A": 24, "I": 25, "O": 31,
		"Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44,
		"c": 45, "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53,
		"l": 54, "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61,
		"t": 62, "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69,
		"ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80,
		"ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92,
		"ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112,
		"ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120,
		"ɹ": 123, "ɾ": 125, "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131,
		"ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140,
		"χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157, "ː": 158,
		"ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177,
	}

	// Load voices.npz
	npzPath := filepath.Join(filepath.Dir(modelPath), "voices.npz")
	if _, err := os.Stat(npzPath); err != nil {
		npzPath = filepath.Join(filepath.Dir(modelPath), "voices.npz")
	}
	if _, err := os.Stat(npzPath); err == nil {
		voices, err := loadNPZVoices(npzPath)
		if err == nil {
			e.voices = voices
		}
	}

	return nil
}

func (e *KittenEngine) Synthesize(text string, voice string, speed float32) ([]float32, int, error) {
	// Friendly voice name to internal file name mapping
	voiceMapping := map[string]string{
		"bella":  "expr-voice-2-f",
		"jasper": "expr-voice-2-m",
		"luna":   "expr-voice-3-f",
		"bruno":  "expr-voice-3-m",
		"rosie":  "expr-voice-4-f",
		"hugo":   "expr-voice-4-m",
		"kiki":   "expr-voice-5-f",
		"leo":    "expr-voice-5-m",
	}

	internalName, isFriendly := voiceMapping[strings.ToLower(voice)]
	var voiceData []float32
	if isFriendly {
		voiceData = e.voices[internalName]
	} else {
		// Try dynamic matching
		for k, vec := range e.voices {
			if strings.Contains(k, strings.ToLower(voice)) || strings.Contains(strings.ToLower(voice), k) {
				voiceData = vec
				break
			}
		}
		if len(voiceData) == 0 {
			// Fallback to Luna
			voiceData = e.voices["expr-voice-3-f"]
			if len(voiceData) == 0 {
				for _, vec := range e.voices {
					voiceData = vec
					break
				}
			}
		}
	}

	tokens := e.tokenize(text)
	if len(tokens) == 0 {
		return nil, e.sampleRate, fmt.Errorf("no tokens mapped from text")
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
		[]string{"input_ids", "style", "speed"},
		[]string{"waveform"},
		nil)
	if err != nil {
		hasSpeed = false
		// Fallback to "input_ids", "style" only
		session, err = ort.NewDynamicAdvancedSession(e.modelPath,
			[]string{"input_ids", "style"},
			[]string{"waveform"},
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
	
	// Trim trailing quiet samples
	var lastNonZero int = len(outputData) - 1
	for lastNonZero >= 0 && outputData[lastNonZero] == 0 {
		lastNonZero--
	}
	if lastNonZero >= 0 {
		outputData = outputData[:lastNonZero+1]
	}

	return outputData, e.sampleRate, nil
}

func (e *KittenEngine) Close() error {
	return nil
}

// Phonemize converts raw English text into IPA phonemes using the local espeak-ng/piper_phonemize subprocess
func Phonemize(text string) ([]string, error) {
	absPiper, _ := filepath.Abs("./piper")
	cmdPath := filepath.Join(absPiper, "piper_phonemize")
	espeakData := filepath.Join(absPiper, "espeak-ng-data-v1-gpl")

	cmd := exec.Command(cmdPath, "-l", "en-us", "--espeak_data", espeakData)
	cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+absPiper)

	var stdin bytes.Buffer
	stdin.WriteString(text + "\n")
	cmd.Stdin = &stdin

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run phonemize: %v, stderr: %s", err, stderr.String())
	}

	var result struct {
		Phonemes []string `json:"phonemes"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal phonemizer output: %v", err)
	}

	return result.Phonemes, nil
}

func (e *KittenEngine) tokenize(text string) []int64 {
	var tokens []int64
	tokens = append(tokens, 0) // Start token

	// Generate IPA phonemes using our robust subprocess phonemizer
	phonemes, err := Phonemize(text)
	if err == nil && len(phonemes) > 0 {
		for _, p := range phonemes {
			if id, ok := e.vocab[p]; ok {
				tokens = append(tokens, id)
			} else {
				// Map individual runes if it's a combined IPA character
				for _, r := range p {
					if id, ok := e.vocab[string(r)]; ok {
						tokens = append(tokens, id)
					}
				}
			}
		}
	} else {
		// Fallback to character mapping if phonemizer fails
		text = strings.ToLower(text)
		for _, r := range text {
			symbol := string(r)
			if id, ok := e.vocab[symbol]; ok {
				tokens = append(tokens, id)
			}
		}
	}

	tokens = append(tokens, 0) // End token
	return tokens
}

func loadNPZVoices(npzPath string) (map[string][]float32, error) {
	r, err := zip.OpenReader(npzPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	voices := make(map[string][]float32)
	for _, f := range r.File {
		if f.FileInfo().IsDir() {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			continue
		}
		data, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			continue
		}

		if len(data) < 12 {
			continue
		}
		// NPY header check
		if !bytes.Equal(data[:6], []byte("\x93NUMPY")) {
			continue
		}
		major := data[6]
		var headerLen int
		var startOffset int
		if major == 1 {
			headerLen = int(binary.LittleEndian.Uint16(data[8:10]))
			startOffset = 10 + headerLen
		} else {
			headerLen = int(binary.LittleEndian.Uint32(data[8:12]))
			startOffset = 12 + headerLen
		}

		if len(data) <= startOffset {
			continue
		}

		floatData := data[startOffset:]
		count := len(floatData) / 4
		vec := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(floatData[i*4 : (i+1)*4])
			vec[i] = math.Float32frombits(bits)
		}

		name := strings.TrimSuffix(f.Name, ".npy")
		voices[name] = vec
	}

	return voices, nil
}
