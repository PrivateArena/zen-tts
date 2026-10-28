package piper

/*
#cgo CFLAGS: -I${SRCDIR}/../../piper/include
#cgo LDFLAGS: -L${SRCDIR}/../../piper -lpiper -Wl,--no-as-needed -lonnxruntime -Wl,--as-needed -Wl,-rpath,${SRCDIR}/../../piper -Wl,-rpath,$ORIGIN/piper -Wl,-rpath,$ORIGIN -Wl,--disable-new-dtags

#include "piper.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type Synthesizer struct {
	ptr *C.piper_synthesizer
}

func NewSynthesizer(modelPath, configPath, espeakDataPath string) (*Synthesizer, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	var cConfigPath *C.char
	if configPath != "" {
		cConfigPath = C.CString(configPath)
		defer C.free(unsafe.Pointer(cConfigPath))
	}

	cEspeakDataPath := C.CString(espeakDataPath)
	defer C.free(unsafe.Pointer(cEspeakDataPath))

	ptr := C.piper_create(cModelPath, cConfigPath, cEspeakDataPath)
	if ptr == nil {
		return nil, fmt.Errorf("failed to create piper synthesizer")
	}

	return &Synthesizer{ptr: ptr}, nil
}

func (s *Synthesizer) Close() {
	if s.ptr != nil {
		C.piper_free(s.ptr)
		s.ptr = nil
	}
}

type Options struct {
	SpeakerID   int
	LengthScale float32
	NoiseScale  float32
	NoiseWScale float32
}

func (s *Synthesizer) DefaultOptions() Options {
	cOpts := C.piper_default_synthesize_options(s.ptr)
	return Options{
		SpeakerID:   int(cOpts.speaker_id),
		LengthScale: float32(cOpts.length_scale),
		NoiseScale:  float32(cOpts.noise_scale),
		NoiseWScale: float32(cOpts.noise_w_scale),
	}
}

func (s *Synthesizer) Synthesize(text string, opts *Options, callback func([]float32, int) bool) error {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var cOpts *C.piper_synthesize_options
	if opts != nil {
		cOpts = (*C.piper_synthesize_options)(C.malloc(C.sizeof_piper_synthesize_options))
		defer C.free(unsafe.Pointer(cOpts))
		cOpts.speaker_id = C.int(opts.SpeakerID)
		cOpts.length_scale = C.float(opts.LengthScale)
		cOpts.noise_scale = C.float(opts.NoiseScale)
		cOpts.noise_w_scale = C.float(opts.NoiseWScale)
	}

	res := C.piper_synthesize_start(s.ptr, cText, cOpts)
	if res != C.PIPER_OK {
		return fmt.Errorf("failed to start synthesis: %d", res)
	}

	var chunk C.piper_audio_chunk
	for {
		res = C.piper_synthesize_next(s.ptr, &chunk)
		if res < C.PIPER_OK {
			return fmt.Errorf("synthesis error: %d", res)
		}

		if chunk.num_samples > 0 {
			// Convert C float array to Go slice without copy
			samples := (*[1 << 30]float32)(unsafe.Pointer(chunk.samples))[:chunk.num_samples:chunk.num_samples]
			if !callback(samples, int(chunk.sample_rate)) {
				break
			}
		}

		if res == C.PIPER_DONE || chunk.is_last {
			break
		}
	}

	return nil
}
