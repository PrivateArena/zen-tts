package kitten

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"zen-tts/internal/shared"

	ort "github.com/yalue/onnxruntime_go"
)

type KittenEngine struct {
	modelPath  string
	voices     map[string][]float32
	sampleRate int
}

func (e *KittenEngine) Initialize(modelPath string, configPath string) error {
	if err := shared.InitONNXRuntime(); err != nil {
		return fmt.Errorf("failed to init onnxruntime: %v", err)
	}

	e.modelPath = modelPath
	e.sampleRate = 24000 // KittenTTS is always 24 kHz
	e.voices = make(map[string][]float32)

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

// Friendly voice name to internal file name mapping
var kittenVoiceMapping = map[string]string{
	"bella":  "expr-voice-2-f",
	"jasper": "expr-voice-2-m",
	"luna":   "expr-voice-3-f",
	"bruno":  "expr-voice-3-m",
	"rosie":  "expr-voice-4-f",
	"hugo":   "expr-voice-4-m",
	"kiki":   "expr-voice-5-f",
	"leo":    "expr-voice-5-m",
}

// resolveVoice maps a requested voice name onto a style vector from voices.npz.
func (e *KittenEngine) resolveVoice(voice string) []float32 {
	internalName, isFriendly := kittenVoiceMapping[strings.ToLower(voice)]
	if isFriendly {
		return e.voices[internalName]
	}
	// Try dynamic matching
	for k, vec := range e.voices {
		if strings.Contains(k, strings.ToLower(voice)) || strings.Contains(strings.ToLower(voice), k) {
			return vec
		}
	}
	// Fallback to Luna
	if vec := e.voices["expr-voice-3-f"]; len(vec) > 0 {
		return vec
	}
	for _, vec := range e.voices {
		return vec
	}
	return nil
}

// styleRow extracts the style embedding row that corresponds to a token sequence
// length, clamped to the number of rows the voice vector carries.
func styleRow(voiceData []float32, numTokens int) []float32 {
	styleVec := make([]float32, 256)
	numEmbeds := len(voiceData) / 256
	if numEmbeds > 0 {
		idx := numTokens
		if idx >= numEmbeds {
			idx = numEmbeds - 1
		}
		copy(styleVec, voiceData[idx*256:(idx+1)*256])
	} else if len(voiceData) >= 256 {
		copy(styleVec, voiceData[:256])
	}
	return styleVec
}

// graphIO is one candidate wiring of the KittenTTS ONNX graph: newer exports take a
// speed input and emit a duration tensor, older ones neither.
type graphIO struct {
	inputs  []string
	outputs []string
}

var kittenGraphIO = []graphIO{
	{[]string{"input_ids", "style", "speed"}, []string{"waveform", "duration"}},
	{[]string{"input_ids", "style", "speed"}, []string{"waveform"}},
	{[]string{"input_ids", "style"}, []string{"waveform", "duration"}},
	{[]string{"input_ids", "style"}, []string{"waveform"}},
}

// synthesize runs the graph once. It returns the raw waveform, the per-token duration
// tensor when the graph exposes one (nil otherwise) and the token stream it was fed, so
// that the audio and its alignment can never come from different passes.
func (e *KittenEngine) synthesize(text string, voice string, speed float32) ([]float32, []int64, *shared.TokenStream, error) {
	voiceData := e.resolveVoice(voice)

	ts := shared.BuildTokenStream(text, "en-us")
	tokens := ts.IDs
	if len(tokens) == 0 {
		return nil, nil, ts, fmt.Errorf("no tokens mapped from text")
	}

	styleVec := styleRow(voiceData, len(tokens))

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

	var lastErr error
	for _, wiring := range kittenGraphIO {
		inputs := []ort.Value{tokensTensor, styleTensor}
		if len(wiring.inputs) == 3 {
			inputs = append(inputs, speedTensor)
		}

		session, err := ort.NewDynamicAdvancedSession(e.modelPath, wiring.inputs, wiring.outputs, nil)
		if err != nil {
			lastErr = err
			continue
		}

		outputs := make([]ort.Value, len(wiring.outputs))
		err = session.Run(inputs, outputs)
		if err != nil {
			session.Destroy()
			lastErr = err
			continue
		}

		wave, durations, castErr := readKittenOutputs(outputs)
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
		session.Destroy()
		if castErr != nil {
			lastErr = castErr
			continue
		}
		return wave, durations, ts, nil
	}

	return nil, nil, ts, fmt.Errorf("failed to run kitten graph: %v", lastErr)
}

func readKittenOutputs(outputs []ort.Value) (wave []float32, durations []int64, err error) {
	for _, out := range outputs {
		if out == nil {
			return nil, nil, fmt.Errorf("dynamic session did not allocate output tensor")
		}
	}
	waveTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("failed to cast output tensor to float32")
	}
	wave = waveTensor.GetData()
	if len(outputs) > 1 {
		durTensor, ok := outputs[1].(*ort.Tensor[int64])
		if !ok {
			return wave, nil, fmt.Errorf("failed to cast duration tensor to int64")
		}
		durations = durTensor.GetData()
	}
	return wave, durations, nil
}

// trimTrailingSilence drops the exact-zero samples the graph pads the end with. It is
// the same trim the audio path has always applied, so the WAV bytes are unchanged.
func trimTrailingSilence(data []float32) []float32 {
	lastNonZero := len(data) - 1
	for lastNonZero >= 0 && data[lastNonZero] == 0 {
		lastNonZero--
	}
	if lastNonZero >= 0 {
		return data[:lastNonZero+1]
	}
	return data
}

func (e *KittenEngine) Synthesize(text string, voice string, speed float32) ([]float32, int, error) {
	wave, _, _, err := e.synthesize(text, voice, speed)
	if err != nil {
		return nil, 0, err
	}
	return trimTrailingSilence(wave), e.sampleRate, nil
}

// SynthesizeWithTimings implements TimingEngine with the model's own duration
// predictor: exact per-word boundaries instead of an estimate, from one synthesis pass.
func (e *KittenEngine) SynthesizeWithTimings(text, voice string, speed float32) ([]float32, []shared.WordTiming, int, error) {
	a, err := e.SynthesizeAligned(text, voice, speed)
	if err != nil {
		return nil, nil, 0, err
	}
	return a.Samples, a.Words, a.SampleRate, nil
}

// SynthesizeAligned implements the exact-timing engine contract. KittenTTS' ONNX graph
// outputs a "duration" tensor next to the waveform, one int64 frame count per token,
// and sum(duration)*600 == number of waveform samples, so these timings are frame-exact.
func (e *KittenEngine) SynthesizeAligned(text, voice string, speed float32) (shared.Alignment, error) {
	wave, durations, ts, err := e.synthesize(text, voice, speed)
	if err != nil {
		return shared.Alignment{}, err
	}
	samples := trimTrailingSilence(wave)

	alignment := shared.Alignment{
		Samples:    samples,
		SampleRate: e.sampleRate,
		Source:     shared.TimingSourceVAD,
		// Overwritten with the model frame size as soon as the duration tensor is used.
		FrameMS: float64(shared.VADWindowMS),
	}
	if len(durations) == 0 {
		// Older graph without a duration output: fall back to silence analysis.
		boundaries, ok := ts.WordBoundaries()
		if !ok {
			boundaries = shared.TokenizeWordBoundaries(text, "en-us")
		}
		alignment.Words = shared.SilenceBoundaryTimings(samples, e.sampleRate, boundaries)
		return alignment, nil
	}

	words, phonemes, frameMS, err := shared.TimingsFromDurations(durations, ts, e.sampleRate, len(wave), len(samples))
	if err != nil {
		boundaries, ok := ts.WordBoundaries()
		if !ok {
			boundaries = shared.TokenizeWordBoundaries(text, "en-us")
		}
		alignment.Words = shared.SilenceBoundaryTimings(samples, e.sampleRate, boundaries)
		return alignment, nil
	}
	alignment.Words = words
	alignment.Phonemes = phonemes
	alignment.Source = shared.TimingSourceDuration
	alignment.FrameMS = frameMS
	return alignment, nil
}

func (e *KittenEngine) SampleRate() int {
	return e.sampleRate
}

func (e *KittenEngine) Close() error {
	return nil
}

func (e *KittenEngine) SynthesizeStream(text string, voice string, speed float32, callback func(samples []float32) bool) error {
	samples, _, err := e.Synthesize(text, voice, speed)
	if err != nil {
		return err
	}
	callback(samples)
	return nil
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
