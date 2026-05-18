package kitten

import (
	"encoding/json"
	"fmt"
	"os"
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
	sampleRate int
}

func (e *KittenEngine) Initialize(modelPath string, configPath string) error {
	if err := InitONNXRuntime(); err != nil {
		return fmt.Errorf("failed to init onnxruntime: %v", err)
	}

	e.modelPath = modelPath
	e.sampleRate = 24000 // KittenTTS is always 24 kHz

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

	return nil
}

func (e *KittenEngine) Synthesize(text string, voice string, speed float32) ([]float32, int, error) {
	// Map default KittenTTS voices to style integer IDs (0 to 7)
	voiceMap := map[string]int64{
		"bella":  0,
		"jasper": 1,
		"luna":   2,
		"bruno":  3,
		"rosie":  4,
		"hugo":   5,
		"kiki":   6,
		"leo":    7,
	}

	voiceID, ok := voiceMap[strings.ToLower(voice)]
	if !ok {
		// Fallback to Luna
		voiceID = 2
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

	// KittenTTS models typically take speaker_id/style as an int64 scalar or 1D tensor
	styleShape := ort.NewShape(1)
	styleTensor, err := ort.NewTensor(styleShape, []int64{voiceID})
	if err != nil {
		return nil, 0, err
	}
	defer styleTensor.Destroy()

	outputShape := ort.NewShape(1, 240000)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, 0, err
	}
	defer outputTensor.Destroy()

	// Re-bind session with the new dynamic shapes
	var session *ort.AdvancedSession
	speedShape := ort.NewShape(1)
	speedTensor, err := ort.NewTensor(speedShape, []float32{speed})
	if err == nil {
		session, err = ort.NewAdvancedSession(e.modelPath,
			[]string{"tokens", "speaker_id", "speed"},
			[]string{"audio"},
			[]ort.ArbitraryTensor{tokensTensor, styleTensor, speedTensor},
			[]ort.ArbitraryTensor{outputTensor},
			nil)
		defer speedTensor.Destroy()
	}

	if err != nil {
		// Fallback to standard input names "tokens", "style"
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
