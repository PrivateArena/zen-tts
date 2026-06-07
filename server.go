package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"zen-tts/internal/kitten"
	"zen-tts/internal/kokoro"
	"zen-tts/internal/piper"
)

var (
	ServerActive bool
	ServerPort   int
	ActiveModel  string
	serverSrv    *http.Server
	serverMu     sync.Mutex
)

// --- SERVER CONTROL ---

func StartServer(modelName string, port int, cpuCore int) {
	serverMu.Lock()
	defer serverMu.Unlock()

	if ServerActive {
		return
	}

	LogMsg(fmt.Sprintf("[yellow]Initializing engine for %s...[-]", modelName))

	onnxPath, configPath := getModelPaths(modelName)
	if onnxPath == "" {
		LogMsg("[red]Failed to find model files[-]")
		return
	}

	var engine TTSEngine
	var err error

	if strings.HasPrefix(modelName, "kokoro") {
		engine = &kokoro.KokoroEngine{}
		err = engine.Initialize(onnxPath, configPath)
	} else if strings.HasPrefix(modelName, "kitten") {
		engine = &kitten.KittenEngine{}
		err = engine.Initialize(onnxPath, configPath)
	} else {
		engine = &piper.PiperEngine{}
		err = engine.Initialize(onnxPath, configPath)
	}

	if err != nil {
		LogMsg(fmt.Sprintf("[red]Engine Initialization Error: %v[-]", err))
		return
	}

	activeSynthMu.Lock()
	if activeSynth != nil {
		activeSynth.Close()
	}
	activeSynth = engine
	activeSynthMu.Unlock()

	ActiveModel = modelName
	ServerPort = port

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" || r.URL.Path == "/tts" {
			ttsHandler(w, r)
		} else {
			http.NotFound(w, r)
		}
	})

	serverSrv = &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}

	go func() {
		LogMsg(fmt.Sprintf("[green]Server started on :%d using %s[-]", port, modelName))
		if err := serverSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			LogMsg(fmt.Sprintf("[red]Server Error: %v[-]", err))
			ServerActive = false
		}
	}()
	ServerActive = true
}

func StopServer() {
	serverMu.Lock()
	defer serverMu.Unlock()

	if !ServerActive || serverSrv == nil {
		return
	}

	LogMsg("Stopping server...")
	serverSrv.Close()

	activeSynthMu.Lock()
	if activeSynth != nil {
		activeSynth.Close()
		activeSynth = nil
	}
	activeSynthMu.Unlock()

	ServerActive = false
	serverSrv = nil
	LogMsg("[yellow]Server stopped[-]")
}

func ToggleServer(model string, port int, cpuCore int) {
	if ServerActive {
		StopServer()
	} else {
		StartServer(model, port, cpuCore)
	}
}

// --- HTTP HANDLER ---

func ttsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "405 Method Not Allowed", 405)
		return
	}

	var req TTSRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad JSON", 400)
		return
	}

	text := normalizeText(req.Text)
	if text == "" {
		http.Error(w, "Empty text", 400)
		return
	}

	userSpeed := req.Speed
	if userSpeed <= 0 {
		userSpeed = 1.0
	}

	LogMsg(fmt.Sprintf("Processing %d chars | Speed: %.1fx | Voice: %s", len(text), userSpeed, req.Voice))

	activeSynthMu.Lock()
	synth := activeSynth
	activeSynthMu.Unlock()

	if synth == nil {
		http.Error(w, "Engine not initialized", 500)
		return
	}

	// ?stream=true — stream WAV audio chunks to the client as they are synthesized
	if r.URL.Query().Get("stream") == "true" || req.Stream {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported by this server/connection", 500)
			return
		}

		sampleRate := synth.SampleRate()

		w.Header().Set("Content-Type", "audio/wav")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Transfer-Encoding", "chunked")

		// Write a fake large WAV header (infinite WAV) so the client plays it continuously
		const fakeSize = 0x7f000000
		binary.Write(w, binary.LittleEndian, []byte("RIFF"))
		binary.Write(w, binary.LittleEndian, uint32(36+fakeSize))
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
		binary.Write(w, binary.LittleEndian, uint32(fakeSize))
		flusher.Flush()

		// Helper to send PCM chunks
		callback := func(samples []float32) bool {
			if len(samples) == 0 {
				return true
			}
			chunkData := make([]byte, 0, len(samples)*2)
			for _, s := range samples {
				val := int16(s * 32767.0)
				chunkData = append(chunkData, byte(val&0xff), byte(val>>8))
			}
			_, err := w.Write(chunkData)
			if err != nil {
				// Stop synthesis if the client disconnected
				return false
			}
			flusher.Flush()
			return true
		}

		var err error
		if streamEngine, ok := synth.(StreamingEngine); ok {
			err = streamEngine.SynthesizeStream(text, req.Voice, float32(userSpeed), callback)
		} else {
			// Fallback: run full synthesis and stream it in one chunk
			var samples []float32
			samples, _, err = synth.Synthesize(text, req.Voice, float32(userSpeed))
			if err == nil {
				callback(samples)
			}
		}

		if err != nil {
			LogMsg(fmt.Sprintf("[red]Streaming Synthesize Error: %v[-]", err))
		}
		return
	}

	// Synthesize using decoupled engine
	startTime := time.Now()
	samples, sampleRate, err := synth.Synthesize(text, req.Voice, float32(userSpeed))
	processTime := time.Since(startTime)
	if err != nil {
		LogMsg(fmt.Sprintf("[red]Synthesize Error: %v[-]", err))
		http.Error(w, "Generation Failed", 500)
		return
	}

	audioDuration := float64(len(samples)) / float64(sampleRate)
	LogMsg(fmt.Sprintf("[green]Synthesized in %v | Playback Duration: %.2fs[-]", processTime, audioDuration))

	// Convert float32 samples to int16 PCM
	audioData := make([]byte, 0, len(samples)*2)
	for _, s := range samples {
		val := int16(s * 32767.0)
		audioData = append(audioData, byte(val&0xff), byte(val>>8))
	}

	if r.URL.Query().Get("play") == "true" || req.Play {
		doneChan := playAudio(audioData, sampleRate)
		<-doneChan
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Write([]byte(`{"status":"completed"}`))
		return
	}

	// ?timestamps=1 — return JSON envelope with base64 WAV + word timings
	if r.URL.Query().Get("timestamps") == "1" {
		var wavBuf []byte
		wavBuf = append(wavBuf, makeWavHeader(sampleRate, len(audioData))...)
		wavBuf = append(wavBuf, audioData...)

		type timingPayload struct {
			Audio    string      `json:"audio"`
			Timings  interface{} `json:"timings"`
			Duration float64     `json:"duration"`
		}
		payload := timingPayload{
			Audio:    base64.StdEncoding.EncodeToString(wavBuf),
			Duration: audioDuration,
		}

		// Use TimingEngine if the active synth supports it
		if te, ok := synth.(TimingEngine); ok {
			_, timings, _, terr := te.SynthesizeWithTimings(text, req.Voice, float32(userSpeed))
			if terr == nil {
				payload.Timings = timings
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		json.NewEncoder(w).Encode(payload)
		return
	}

	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	writeWavHeader(w, sampleRate, len(audioData))
	w.Write(audioData)
}
