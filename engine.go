package main

import (
	"encoding/binary"
	"io"
	"sync"
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

	// Close tears down CGO memory footprints and terminates runtime instances safely.
	Close() error
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
