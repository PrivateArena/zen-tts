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

	// Load vocab from configPath
	if configPath != "" {
		cfgData, err := os.ReadFile(configPath)
		if err == nil {
			var configMap map[string]interface{}
			if err := json.Unmarshal(cfgData, &configMap); err == nil {
				if vocabVal, ok := configMap["vocab"]; ok {
					if vocabMap, ok := vocabVal.(map[string]interface{}); ok {
						e.vocab = make(map[string]int64)
						for k, v := range vocabMap {
							if valFloat, ok := v.(float64); ok {
								e.vocab[k] = int64(valFloat)
							}
						}
					}
				}
			}
		}
	}

	// Default fallback vocab for KittenTTS (similar to espeak standard)
	if len(e.vocab) == 0 {
		e.vocab = make(map[string]int64)
		chars := " ;:,.!?-' abcdefghijklmnopqrstuvwxyzæçðøħŋœɔɟɥɨɪʝɭɬɱɯɰɲɳɴɵɸɹɺɻɽɾʀʁʂʃʄʈθʉʊʋʌɣɤʍχʎʏʐʑʒʔʕ"
		for i, c := range chars {
			e.vocab[string(c)] = int64(i)
		}
	}

	// Load voices.npz
	npzPath := filepath.Join(filepath.Dir(modelPath), "voices.npz")
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

func (e *KittenEngine) tokenize(text string) []int64 {
	var tokens []int64
	tokens = append(tokens, 0) // Start token

	text = strings.ToLower(text)
	for _, r := range text {
		symbol := string(r)
		if id, ok := e.vocab[symbol]; ok {
			tokens = append(tokens, id)
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
