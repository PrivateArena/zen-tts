package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	"github.com/ebitengine/oto/v3"
)

// --- REAL-TIME AUDIO PLAYBACK (oto) ---

type playRequest struct {
	data       []byte
	sampleRate int
}

var (
	playChan     chan playRequest
	playOnce     sync.Once
	playMu       sync.Mutex
	currentCtx   *oto.Context
	activePlayer *oto.Player
)

// resampleMonoInt16 resamples signed 16-bit LE mono PCM data from srcRate to dstRate.
func resampleMonoInt16(input []byte, srcRate, dstRate int) []byte {
	if srcRate == dstRate || len(input) < 2 {
		return input
	}

	srcSamples := make([]int16, len(input)/2)
	for i := 0; i < len(srcSamples); i++ {
		srcSamples[i] = int16(binary.LittleEndian.Uint16(input[i*2:]))
	}

	ratio := float64(srcRate) / float64(dstRate)
	dstLen := int(float64(len(srcSamples)) / ratio)
	dstSamples := make([]int16, dstLen)

	for i := 0; i < dstLen; i++ {
		srcIdx := float64(i) * ratio
		low := int(srcIdx)
		high := low + 1
		if high >= len(srcSamples) {
			high = len(srcSamples) - 1
		}
		weight := srcIdx - float64(low)

		val := float64(srcSamples[low])*(1.0-weight) + float64(srcSamples[high])*weight
		dstSamples[i] = int16(val)
	}

	output := make([]byte, len(dstSamples)*2)
	for i, s := range dstSamples {
		binary.LittleEndian.PutUint16(output[i*2:], uint16(s))
	}
	return output
}

func audioPlaybackWorker() {
	for req := range playChan {
		playData(req.data, req.sampleRate)
	}
}

func playData(data []byte, sampleRate int) {
	playMu.Lock()
	if currentCtx == nil {
		opts := &oto.NewContextOptions{
			SampleRate:   24000,
			ChannelCount: 1,
			Format:       oto.FormatSignedInt16LE,
		}
		var readyChan chan struct{}
		var err error
		currentCtx, readyChan, err = oto.NewContext(opts)
		if err != nil {
			LogMsg(fmt.Sprintf("[red]Failed to create oto context: %v[-]", err))
			playMu.Unlock()
			return
		}
		<-readyChan
	}
	playMu.Unlock()

	// Resample mono audio to 24000 Hz if necessary
	if sampleRate != 24000 {
		data = resampleMonoInt16(data, sampleRate, 24000)
	}

	playMu.Lock()
	p := currentCtx.NewPlayer(bytes.NewReader(data))
	activePlayer = p
	p.Play()
	playMu.Unlock()

	for {
		playMu.Lock()
		if activePlayer != p {
			playMu.Unlock()
			break
		}
		playing := p.IsPlaying()
		playMu.Unlock()

		if !playing {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	playMu.Lock()
	if activePlayer == p {
		p.Close()
		activePlayer = nil
	}
	playMu.Unlock()
}

func playAudio(data []byte, sampleRate int) {
	playOnce.Do(func() {
		playChan = make(chan playRequest, 16)
		go audioPlaybackWorker()
	})

	playMu.Lock()
	if activePlayer != nil {
		activePlayer.Close()
		activePlayer = nil
	}
	playMu.Unlock()

	dataCopy := make([]byte, len(data))
	copy(dataCopy, data)

	select {
	case playChan <- playRequest{data: dataCopy, sampleRate: sampleRate}:
	default:
		LogMsg("[yellow]Audio playback queue full, skipping...[-]")
	}
}
