package inflect

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"sync"

	"zen-tts/internal/shared"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ErrEmptyText     = fmt.Errorf("empty text after normalization")
	ErrEngineClosed  = fmt.Errorf("inflect engine is closed")
	defaultVariation = float32(0.667)
)

type InflectEngine struct {
	modelDir     string
	durationSess *ort.DynamicAdvancedSession
	decodeSess   *ort.DynamicAdvancedSession
	sampleRate   int
	mu           sync.Mutex
	vocab        map[string]int64
	closed       bool
}

func (e *InflectEngine) Initialize(modelDir, configPath string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := shared.InitONNXRuntime(); err != nil {
		return fmt.Errorf("failed to init onnxruntime: %v", err)
	}

	e.vocab = make(map[string]int64, len(symbolToID))
	for k, v := range symbolToID {
		e.vocab[k] = v
	}

	durationPath := filepath.Join(modelDir, "duration.onnx")
	decodePath := filepath.Join(modelDir, "decode.onnx")

	var err error
	e.durationSess, err = ort.NewDynamicAdvancedSession(durationPath,
		[]string{"tokens", "lengths", "length_scale"},
		[]string{"m_p_exp", "logs_p_exp", "y_mask"}, nil)
	if err != nil {
		return fmt.Errorf("duration session: %w", err)
	}

	e.decodeSess, err = ort.NewDynamicAdvancedSession(decodePath,
		[]string{"m_p_exp", "logs_p_exp", "y_mask", "zp_noise", "noise_scale"},
		[]string{"waveform"}, nil)
	if err != nil {
		e.durationSess.Destroy()
		e.durationSess = nil
		return fmt.Errorf("decode session: %w", err)
	}

	e.sampleRate = 24000
	e.modelDir = modelDir
	e.closed = false
	return nil
}

func (e *InflectEngine) Synthesize(text, voice string, speed float32) ([]float32, int, error) {
	return e.SynthesizeWithParams(text, voice, speed, nil)
}

func (e *InflectEngine) SynthesizeStream(text, voice string, speed float32, callback func([]float32) bool) error {
	return e.SynthesizeStreamWithParams(text, voice, speed, nil, callback)
}

func (e *InflectEngine) SynthesizeWithParams(text, voice string, speed float32, params map[string]any) ([]float32, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil, 0, ErrEngineClosed
	}

	seed := int64(0)
	if v, ok := params["seed"].(int64); ok {
		seed = v
	}
	variation := defaultVariation
	if v, ok := params["variation"].(float32); ok {
		variation = v
	} else if v, ok := params["variation"].(float64); ok {
		variation = float32(v)
	}

	return e.synthesizeChunks(text, speed, variation, seed)
}

func (e *InflectEngine) SynthesizeStreamWithParams(text, voice string, speed float32, params map[string]any, callback func([]float32) bool) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return ErrEngineClosed
	}

	seed := int64(0)
	if v, ok := params["seed"].(int64); ok {
		seed = v
	}
	variation := defaultVariation
	if v, ok := params["variation"].(float32); ok {
		variation = v
	} else if v, ok := params["variation"].(float64); ok {
		variation = float32(v)
	}

	return e.streamChunks(text, speed, variation, seed, callback)
}

func (e *InflectEngine) synthesizeChunks(text string, speed, variation float32, baseSeed int64) ([]float32, int, error) {
	chunks := SplitText(text, 200)
	if len(chunks) == 0 {
		return nil, 0, ErrEmptyText
	}

	norm := NormalizeText(text)
	if norm == "" {
		return nil, 0, ErrEmptyText
	}

	var allSamples []float32
	for i, chunk := range chunks {
		if i > 0 {
			pause := BoundaryPauseSeconds(chunks[i-1])
			silence := make([]float32, int(pause*float64(e.sampleRate)))
			allSamples = append(allSamples, silence...)
		}
		wav, err := e.synthesizeChunk(chunk, speed, variation, baseSeed+int64(i))
		if err != nil {
			return nil, 0, err
		}
		allSamples = append(allSamples, EdgeFade(wav, e.sampleRate, 10)...)
	}
	return allSamples, e.sampleRate, nil
}

func (e *InflectEngine) streamChunks(text string, speed, variation float32, baseSeed int64, callback func([]float32) bool) error {
	chunks := SplitText(text, 200)
	if len(chunks) == 0 {
		return nil
	}

	for i, chunk := range chunks {
		if i > 0 {
			pause := BoundaryPauseSeconds(chunks[i-1])
			silence := make([]float32, int(pause*float64(e.sampleRate)))
			if !callback(silence) {
				return nil
			}
		}
		wav, err := e.synthesizeChunk(chunk, speed, variation, baseSeed+int64(i))
		if err != nil {
			return err
		}
		faded := EdgeFade(wav, e.sampleRate, 10)
		if !callback(faded) {
			return nil
		}
	}
	return nil
}

func (e *InflectEngine) synthesizeChunk(text string, speed, variation float32, seed int64) ([]float32, error) {
	norm := NormalizeText(text)
	if norm == "" {
		return nil, nil
	}

	phonemes, err := shared.Phonemize(norm, "en-us")
	if err != nil {
		return nil, fmt.Errorf("phonemize failed: %w", err)
	}
	if len(phonemes) == 0 {
		return nil, nil
	}

	tokens := PhonemesToTokens(phonemes, e.vocab)
	if len(tokens) == 0 {
		return nil, nil
	}

	tokensShape := ort.NewShape(1, int64(len(tokens)))
	tokensTensor, err := ort.NewTensor(tokensShape, tokens)
	if err != nil {
		return nil, fmt.Errorf("tokens tensor: %w", err)
	}
	defer tokensTensor.Destroy()

	lengthsTensor, err := ort.NewTensor(ort.NewShape(1), []int64{int64(len(tokens))})
	if err != nil {
		return nil, fmt.Errorf("lengths tensor: %w", err)
	}
	defer lengthsTensor.Destroy()

	lengthScale := float32(1.0 / speed)
	if lengthScale < 0.5 {
		lengthScale = 0.5
	}
	scaleTensor, err := ort.NewTensor(ort.NewShape(1), []float32{lengthScale})
	if err != nil {
		return nil, fmt.Errorf("scale tensor: %w", err)
	}
	defer scaleTensor.Destroy()

	durationOutputs := []ort.Value{nil, nil, nil}
	err = e.durationSess.Run([]ort.Value{tokensTensor, lengthsTensor, scaleTensor}, durationOutputs)
	if err != nil {
		return nil, fmt.Errorf("duration run: %w", err)
	}
	m_p_exp := durationOutputs[0].(*ort.Tensor[float32])
	logs_p_exp := durationOutputs[1].(*ort.Tensor[float32])
	y_mask := durationOutputs[2].(*ort.Tensor[float32])
	defer m_p_exp.Destroy()
	defer logs_p_exp.Destroy()
	defer y_mask.Destroy()

	mData := m_p_exp.GetData()
	var noise []float32
	if variation == 0 {
		noise = make([]float32, len(mData))
	} else {
		rng := rand.New(rand.NewSource(seed))
		noise = make([]float32, len(mData))
		for i := range noise {
			noise[i] = float32(rng.NormFloat64())
		}
	}

	noiseShape := ort.NewShape(int64(len(mData)))
	noiseTensor, err := ort.NewTensor(noiseShape, noise)
	if err != nil {
		return nil, fmt.Errorf("noise tensor: %w", err)
	}
	defer noiseTensor.Destroy()

	varScaleTensor, err := ort.NewTensor(ort.NewShape(1), []float32{variation})
	if err != nil {
		return nil, fmt.Errorf("variation tensor: %w", err)
	}
	defer varScaleTensor.Destroy()

	decodeOutputs := []ort.Value{nil}
	err = e.decodeSess.Run([]ort.Value{m_p_exp, logs_p_exp, y_mask, noiseTensor, varScaleTensor}, decodeOutputs)
	if err != nil {
		return nil, fmt.Errorf("decode run: %w", err)
	}

	wavTensor := decodeOutputs[0].(*ort.Tensor[float32])
	defer wavTensor.Destroy()

	return wavTensor.GetData(), nil
}

func (e *InflectEngine) SampleRate() int {
	return e.sampleRate
}

func (e *InflectEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}
	e.closed = true

	if e.durationSess != nil {
		e.durationSess.Destroy()
		e.durationSess = nil
	}
	if e.decodeSess != nil {
		e.decodeSess.Destroy()
		e.decodeSess = nil
	}
	return nil
}
