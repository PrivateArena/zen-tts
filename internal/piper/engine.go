package piper

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"zen-tts/internal/shared"
)

type PiperEngine struct {
	synth      *Synthesizer
	sampleRate int
	mu         sync.Mutex
}

func (e *PiperEngine) Initialize(modelPath string, configPath string) error {
	absPiper, _ := filepath.Abs("./piper")
	espeakData := filepath.Join(absPiper, "espeak-ng-data-v1-gpl")

	synth, err := NewSynthesizer(modelPath, configPath, espeakData)
	if err != nil {
		return err
	}
	e.synth = synth
	e.sampleRate = getPiperSampleRate(configPath)
	return nil
}

func (e *PiperEngine) Synthesize(text string, voice string, speed float32) ([]float32, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.synth == nil {
		return nil, 0, fmt.Errorf("piper engine not initialized")
	}
	if speed <= 0 {
		speed = 1.0
	}
	lengthScale := 1.0 / speed
	opts := e.synth.DefaultOptions()
	opts.LengthScale = lengthScale

	var allSamples []float32
	err := e.synth.Synthesize(text, &opts, func(samples []float32, rate int) bool {
		allSamples = append(allSamples, samples...)
		return true
	})
	if err != nil {
		return nil, 0, err
	}
	return allSamples, e.sampleRate, nil
}

func (e *PiperEngine) SynthesizeStream(text string, voice string, speed float32, callback func(samples []float32) bool) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.synth == nil {
		return fmt.Errorf("piper engine not initialized")
	}
	if speed <= 0 {
		speed = 1.0
	}
	lengthScale := 1.0 / speed
	opts := e.synth.DefaultOptions()
	opts.LengthScale = lengthScale

	return e.synth.Synthesize(text, &opts, func(samples []float32, rate int) bool {
		return callback(samples)
	})
}

func (e *PiperEngine) SampleRate() int {
	return e.sampleRate
}
func (e *PiperEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.synth != nil {
		e.synth.Close()
		e.synth = nil
	}
	return nil
}

// SynthesizeWithTimings implements the TimingEngine contract. Piper's ONNX exports do
// not return the VITS duration tensor (monotonic alignment search runs only at training
// time), so the per-word timestamps come from silence analysis of the produced PCM.
func (e *PiperEngine) SynthesizeWithTimings(text, voice string, speed float32) ([]float32, []shared.WordTiming, int, error) {
	alignment, err := e.SynthesizeAligned(text, voice, speed)
	if err != nil {
		return nil, nil, 0, err
	}
	return alignment.Samples, alignment.Words, alignment.SampleRate, nil
}

// SynthesizeAligned synthesizes once and estimates word timings from the silence
// structure of the audio. The token-count prior is character count, because the
// engine's own phoneme inventory is not visible from here.
func (e *PiperEngine) SynthesizeAligned(text, voice string, speed float32) (shared.Alignment, error) {
	samples, sampleRate, err := e.Synthesize(text, voice, speed)
	if err != nil {
		return shared.Alignment{}, err
	}
	alignment := shared.Alignment{
		Samples:    samples,
		SampleRate: sampleRate,
		Source:     shared.TimingSourceVAD,
		FrameMS:    float64(shared.VADWindowMS), // the analysis window the estimate is quantized to
	}
	alignment.Words = shared.SilenceBoundaryTimings(samples, sampleRate, shared.CharCountBoundaries(text))
	return alignment, nil
}

func getPiperSampleRate(path string) int {
	f, err := os.ReadFile(path)
	if err != nil {
		return 22050
	}
	var d struct {
		Audio struct {
			SampleRate int `json:"sample_rate"`
		} `json:"audio"`
	}
	json.Unmarshal(f, &d)
	if d.Audio.SampleRate == 0 {
		return 22050
	}
	return d.Audio.SampleRate
}
