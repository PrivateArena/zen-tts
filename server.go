package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

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
	mux.HandleFunc("/tts", ttsHandler)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tts" {
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

	// Synthesize using decoupled engine
	samples, sampleRate, err := synth.Synthesize(text, req.Voice, float32(userSpeed))
	if err != nil {
		LogMsg(fmt.Sprintf("[red]Synthesize Error: %v[-]", err))
		http.Error(w, "Generation Failed", 500)
		return
	}

	// Convert float32 samples to int16 PCM
	audioData := make([]byte, 0, len(samples)*2)
	for _, s := range samples {
		val := int16(s * 32767.0)
		audioData = append(audioData, byte(val&0xff), byte(val>>8))
	}

	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	writeWavHeader(w, sampleRate, len(audioData))
	w.Write(audioData)
}
