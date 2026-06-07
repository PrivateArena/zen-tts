package main

import (
	"bytes"
	"encoding/binary"
	"io"
	"sync"

	"zen-tts/internal/shared"
)

var (
	activeSynth   TTSEngine
	activeSynthMu sync.Mutex
)

// TTSEngine provides a unified contract for text-to-speech backends.
type TTSEngine interface {
	// Initialize boots the ONNX sessions, binds allocator symbols, and loads voice profiles.
	Initialize(modelPath string, configPath string) error

	// Synthesize converts text to raw PCM 32-bit float or 16-bit int data.
	Synthesize(text string, voice string, speed float32) ([]float32, int, error)

	// SampleRate returns the sample rate of the synthesizer engine.
	SampleRate() int

	// Close tears down CGO memory footprints and terminates runtime instances safely.
	Close() error
}

// TimingEngine is an optional upgrade to TTSEngine. Engines that can produce
// ground-truth word timings via PCM analysis implement this interface.
// Use type assertion at call sites: if te, ok := synth.(TimingEngine); ok { ... }
type TimingEngine interface {
	TTSEngine
	// SynthesizeWithTimings returns samples plus per-word timings relative to segment start.
	SynthesizeWithTimings(text, voice string, speed float32) ([]float32, []shared.WordTiming, int, error)
}

// StreamingEngine is an optional upgrade to TTSEngine. Engines that support
// chunked streaming synthesis implement this interface.
type StreamingEngine interface {
	TTSEngine
	SynthesizeStream(text string, voice string, speed float32, callback func(samples []float32) bool) error
}


func writeWavHeader(w io.Writer, sampleRate int, dataSize int) {
	binary.Write(w, binary.LittleEndian, []byte("RIFF"))
	binary.Write(w, binary.LittleEndian, uint32(36+dataSize))
	binary.Write(w, binary.LittleEndian, []byte("WAVE"))
	binary.Write(w, binary.LittleEndian, []byte("fmt "))
	binary.Write(w, binary.LittleEndian, uint32(16))
	binary.Write(w, binary.LittleEndian, uint16(1))
	binary.Write(w, binary.LittleEndian, uint16(1))
	binary.Write(w, binary.LittleEndian, uint32(sampleRate))
	binary.Write(w, binary.LittleEndian, uint32(sampleRate*2))
	binary.Write(w, binary.LittleEndian, uint16(2))
	binary.Write(w, binary.LittleEndian, uint16(16))
	binary.Write(w, binary.LittleEndian, []byte("data"))
	binary.Write(w, binary.LittleEndian, uint32(dataSize))
}

// makeWavHeader returns a 44-byte WAV header as []byte for in-memory use.
func makeWavHeader(sampleRate int, dataSize int) []byte {
	var buf bytes.Buffer
	writeWavHeader(&buf, sampleRate, dataSize)
	return buf.Bytes()
}
