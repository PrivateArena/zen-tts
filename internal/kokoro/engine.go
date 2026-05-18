package kokoro

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
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
		// Look for libonnxruntime.so inside piper directory first
		libPath := "./piper/libonnxruntime.so.1.24.2"
		if _, err := os.Stat(libPath); err != nil {
			libPath = "./piper/libonnxruntime.so"
		}
		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

type KokoroEngine struct {
	modelPath  string
	vocab      map[string]int64
	voices     map[string][]float32
	sampleRate int
}

func (e *KokoroEngine) Initialize(modelPath string, configPath string) error {
	if err := InitONNXRuntime(); err != nil {
		return fmt.Errorf("failed to init onnxruntime: %v", err)
	}

	e.modelPath = modelPath
	e.sampleRate = 24000
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

	// Default fallback vocab
	if len(e.vocab) == 0 {
		e.vocab = make(map[string]int64)
		chars := " ;:,.!?-' abcdefghijklmnopqrstuvwxyzæçðøħŋœɔɟɥɨɪʝɭɬɱɯɰɲɳɴɵɸɹɺɻɽɾʀʁʂʃʄʈθʉʊʋʌɣɤʍχʎʏʐʑʒʔʕ"
		for i, c := range chars {
			e.vocab[string(c)] = int64(i)
		}
	}

	// Load voice style vectors
	voicesDir := filepath.Join(filepath.Dir(modelPath), "voices")
	files, err := os.ReadDir(voicesDir)
	if err == nil {
		for _, f := range files {
			if !f.IsDir() && (filepath.Ext(f.Name()) == ".bin" || filepath.Ext(f.Name()) == ".json") {
				vName := strings.TrimSuffix(f.Name(), filepath.Ext(f.Name()))
				vPath := filepath.Join(voicesDir, f.Name())
				vec, err := loadVoiceVector(vPath)
				if err == nil {
					e.voices[strings.ToLower(vName)] = vec
				}
			}
		}
	}

	return nil
}

func (e *KokoroEngine) Synthesize(text string, voice string, speed float32) ([]float32, int, error) {
	styleVec, ok := e.voices[strings.ToLower(voice)]
	if !ok {
		for k, vec := range e.voices {
			if strings.Contains(k, strings.ToLower(voice)) || strings.Contains(strings.ToLower(voice), k) {
				styleVec = vec
				break
			}
		}
		if len(styleVec) == 0 {
			for _, vec := range e.voices {
				styleVec = vec
				break
			}
			if len(styleVec) == 0 {
				styleVec = make([]float32, 256)
			}
		}
	}

	tokens := e.tokenize(text)
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

	// In Kokoro, dynamic output size is used
	outputShape := ort.NewShape(1, 240000) // Up to 10 seconds of output
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, 0, err
	}
	defer outputTensor.Destroy()

	// Try running session with "tokens", "style", "speed" inputs first, then fallback to "tokens", "style"
	var session *ort.AdvancedSession
	speedShape := ort.NewShape(1)
	speedTensor, err := ort.NewTensor(speedShape, []float32{speed})
	if err == nil {
		session, err = ort.NewAdvancedSession(e.modelPath,
			[]string{"tokens", "style", "speed"},
			[]string{"audio"},
			[]ort.ArbitraryTensor{tokensTensor, styleTensor, speedTensor},
			[]ort.ArbitraryTensor{outputTensor},
			nil)
		defer speedTensor.Destroy()
	}

	if err != nil {
		// Fallback to "tokens", "style" only
		session, err = ort.NewAdvancedSession(e.modelPath,
			[]string{"tokens", "style"},
			[]string{"audio"},
			[]ort.ArbitraryTensor{tokensTensor, styleTensor},
			[]ort.ArbitraryTensor{outputTensor},
			nil)
	}

	if err != nil {
		return nil, 0, fmt.Errorf("failed to create session: %v", err)
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		return nil, 0, fmt.Errorf("failed to run session: %v", err)
	}

	outputData := outputTensor.GetData()
	
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

func (e *KokoroEngine) Close() error {
	return nil
}

func (e *KokoroEngine) tokenize(text string) []int64 {
	// Clean text and map each character or known IPA phoneme to its token ID.
	// NOTE: For highly natural speech synthesis, plug in a Grapheme-to-Phoneme (g2p)
	// phonemizer such as espeak-ng or misaki before feeding tokens to the model.
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
		if count > 256 {
			count = 256
		}
		vec = make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
			vec[i] = math.Float32frombits(bits)
		}
		return vec, nil
	}
	return nil, fmt.Errorf("invalid voice style file size/format")
}
