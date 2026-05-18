package kokoro

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
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
				if err == nil {
					e.voices[strings.ToLower(vName)] = vec
				}
			}
		}
	}

	return nil
}

func (e *KokoroEngine) Synthesize(text string, voice string, speed float32) ([]float32, int, error) {

	tokens := shared.Tokenize(text)
	voiceData, ok := e.voices[strings.ToLower(voice)]
	if len(voiceData) == 0 {
		for k, vec := range e.voices {
			if strings.Contains(k, strings.ToLower(voice)) || strings.Contains(strings.ToLower(voice), k) {
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
