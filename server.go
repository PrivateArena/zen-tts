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

	"zen-tts/internal/inflect"
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

// SwitchEngine loads and initializes the requested engine and model, closing the old one.
// Thread-safe and safe to call on-the-fly while requests are being processed.
func SwitchEngine(engineType string, modelName string) error {
	activeSynthMu.Lock()
	defer activeSynthMu.Unlock()

	ConfigMu.RLock()
	currEngine := CurrentConfig.ActiveEngine
	currModel := CurrentConfig.LastModel
	ConfigMu.RUnlock()

	// If the requested engine/model is already active, do nothing
	if activeSynth != nil && currEngine == engineType && currModel == modelName {
		return nil
	}

	LogMsg(fmt.Sprintf("[yellow]Switching engine to %s (%s)...[-]", engineType, modelName))

	var onnxPath, configPath string
	ConfigMu.RLock()
	engCfg, hasEngCfg := CurrentConfig.Engines[engineType]
	ConfigMu.RUnlock()

	if hasEngCfg && engCfg.ModelPath != "" && engCfg.ConfigPath != "" {
		onnxPath = engCfg.ModelPath
		configPath = engCfg.ConfigPath
	} else {
		onnxPath, configPath = getModelPaths(modelName)
	}

	if onnxPath == "" {
		return fmt.Errorf("failed to find model files for model: %s", modelName)
	}

	var engine TTSEngine
	var err error

	if engineType == "inflect" {
		engine = &inflect.InflectEngine{}
		err = engine.Initialize(onnxPath, configPath)
	} else if engineType == "kokoro" {
		engine = &kokoro.KokoroEngine{}
		err = engine.Initialize(onnxPath, configPath)
	} else if engineType == "kitten" {
		engine = &kitten.KittenEngine{}
		err = engine.Initialize(onnxPath, configPath)
	} else {
		engine = &piper.PiperEngine{}
		err = engine.Initialize(onnxPath, configPath)
	}

	if err != nil {
		return fmt.Errorf("engine initialization failed: %v", err)
	}

	if activeSynth != nil {
		activeSynth.Close()
	}
	activeSynth = engine
	ActiveModel = modelName

	ConfigMu.Lock()
	CurrentConfig.ActiveEngine = engineType
	CurrentConfig.LastModel = modelName
	if cfg, ok := CurrentConfig.Engines[engineType]; ok {
		cfg.Model = modelName
		CurrentConfig.Engines[engineType] = cfg
	}
	ConfigMu.Unlock()
	SaveConfig()

	LogMsg(fmt.Sprintf("[green]Engine successfully switched to %s (%s)[-]", engineType, modelName))
	return nil
}

// --- SERVER CONTROL ---

func StartServer(modelName string, port int, cpuCore int) {
	serverMu.Lock()
	defer serverMu.Unlock()

	if ServerActive {
		return
	}

	ConfigMu.RLock()
	activeEngine := CurrentConfig.ActiveEngine
	engCfg, hasEngCfg := CurrentConfig.Engines[activeEngine]
	ConfigMu.RUnlock()

	// Check compatibility between the active engine and the requested model name
	compatible := false
	if activeEngine == "kokoro" && strings.HasPrefix(modelName, "kokoro") {
		compatible = true
	} else if activeEngine == "kitten" && strings.HasPrefix(modelName, "kitten") {
		compatible = true
	} else if activeEngine == "inflect" && strings.HasPrefix(modelName, "inflect") {
		compatible = true
	} else if activeEngine == "piper" && !strings.HasPrefix(modelName, "kokoro") && !strings.HasPrefix(modelName, "kitten") && !strings.HasPrefix(modelName, "inflect") {
		compatible = true
	}

	// If they are not compatible (e.g. user changed active_engine in config.json manually
	// but didn't update last_model), default to the active engine's configured model.
	if !compatible && hasEngCfg && engCfg.Model != "" {
		modelName = engCfg.Model
	}

	var engineType string
	if strings.HasPrefix(modelName, "inflect") {
		engineType = "inflect"
	} else if strings.HasPrefix(modelName, "kokoro") {
		engineType = "kokoro"
	} else if strings.HasPrefix(modelName, "kitten") {
		engineType = "kitten"
	} else {
		engineType = "piper"
	}

	err := SwitchEngine(engineType, modelName)
	if err != nil {
		LogMsg(fmt.Sprintf("[red]Engine Initialization Error: %v[-]", err))
		return
	}

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
		LogMsg(fmt.Sprintf("[green]Server started on :%d[-]", port))
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

	ConfigMu.RLock()
	activeEngine := CurrentConfig.ActiveEngine
	ConfigMu.RUnlock()

	// Detect requested engine from voice or engine field
	targetEngine := activeEngine
	if req.Engine != "" {
		targetEngine = strings.ToLower(req.Engine)
	} else if strings.HasPrefix(req.Voice, "inflect") {
		targetEngine = "inflect"
	} else if strings.HasPrefix(req.Voice, "kokoro") {
		targetEngine = "kokoro"
	} else if strings.HasPrefix(req.Voice, "kitten") {
		targetEngine = "kitten"
	} else if req.Voice != "" {
		if kokoro.KokoroVoices[strings.ToLower(req.Voice)] {
			targetEngine = "kokoro"
		}
	}

	if targetEngine != activeEngine && (targetEngine == "piper" || targetEngine == "kokoro" || targetEngine == "kitten" || targetEngine == "inflect") {
		ConfigMu.RLock()
		targetModel := CurrentConfig.Engines[targetEngine].Model
		ConfigMu.RUnlock()
		err := SwitchEngine(targetEngine, targetModel)
		if err != nil {
			LogMsg(fmt.Sprintf("[red]Failed to switch engine to %s: %v[-]", targetEngine, err))
			http.Error(w, "Engine switch failed", 500)
			return
		}
		activeEngine = targetEngine
	}

	reqVoice := req.Voice
	userSpeed := req.Speed

	ConfigMu.RLock()
	engCfg, hasEngCfg := CurrentConfig.Engines[activeEngine]
	ConfigMu.RUnlock()

	if reqVoice == "" && hasEngCfg {
		reqVoice = engCfg.Voice
	}

	if userSpeed <= 0 {
		if hasEngCfg && engCfg.Speed > 0 {
			userSpeed = engCfg.Speed
		} else {
			userSpeed = 1.0
		}
	}

	LogMsg(fmt.Sprintf("Processing %d chars | Speed: %.1fx | Voice: %s | Engine: %s", len(text), userSpeed, reqVoice, activeEngine))

	activeSynthMu.Lock()
	synth := activeSynth
	activeSynthMu.Unlock()

	if synth == nil {
		http.Error(w, "Engine not initialized", 500)
		return
	}

	// Build extra params map for ParameterizedEngine
	params := map[string]any{}
	if req.Seed != nil {
		params["seed"] = *req.Seed
	}
	if req.Variation != nil {
		params["variation"] = *req.Variation
	}
	hasParams := len(params) > 0

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
		if hasParams {
			if pe, ok := synth.(ParameterizedEngine); ok {
				err = pe.SynthesizeStreamWithParams(text, reqVoice, float32(userSpeed), params, callback)
			} else if streamEngine, ok := synth.(StreamingEngine); ok {
				err = streamEngine.SynthesizeStream(text, reqVoice, float32(userSpeed), callback)
			} else {
				var samples []float32
				samples, _, err = synth.Synthesize(text, reqVoice, float32(userSpeed))
				if err == nil {
					callback(samples)
				}
			}
		} else if streamEngine, ok := synth.(StreamingEngine); ok {
			err = streamEngine.SynthesizeStream(text, reqVoice, float32(userSpeed), callback)
		} else {
			// Fallback: run full synthesis and stream it in one chunk
			var samples []float32
			samples, _, err = synth.Synthesize(text, reqVoice, float32(userSpeed))
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
	var (
		samples    []float32
		sampleRate int
		synthErr   error
	)
	if hasParams {
		if pe, ok := synth.(ParameterizedEngine); ok {
			samples, sampleRate, synthErr = pe.SynthesizeWithParams(text, reqVoice, float32(userSpeed), params)
		} else {
			samples, sampleRate, synthErr = synth.Synthesize(text, reqVoice, float32(userSpeed))
		}
	} else {
		samples, sampleRate, synthErr = synth.Synthesize(text, reqVoice, float32(userSpeed))
	}
	processTime := time.Since(startTime)
	if synthErr != nil {
		LogMsg(fmt.Sprintf("[red]Synthesize Error: %v[-]", synthErr))
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
			_, timings, _, terr := te.SynthesizeWithTimings(text, reqVoice, float32(userSpeed))
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
