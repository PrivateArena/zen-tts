package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"zen-tts/internal/piper"
)

var (
	ServerActive  bool
	ServerPort    int
	ActiveModel   string
	serverSrv     *http.Server
	serverMu      sync.Mutex
	activeSynth   *piper.Synthesizer
	activeSynthMu sync.Mutex
)

// --- TEXT NORMALIZATION ---

func normalizeText(text string) string {
	ConfigMu.RLock()
	defer ConfigMu.RUnlock()

	// 1. Custom Rules
	for _, rule := range CurrentConfig.Replacements {
		if rule.IsRegex {
			if re, ok := RegexCache[rule.Pattern]; ok {
				text = re.ReplaceAllString(text, rule.Replacement)
			}
		} else {
			text = strings.ReplaceAll(text, rule.Pattern, rule.Replacement)
		}
	}

	// 2. Generic Safety (Squash repeats)
	return squashRepeats(text, 3)
}

func squashRepeats(s string, maxRepeats int) string {
	if len(s) == 0 { return s }
	var b strings.Builder
	runes := []rune(s)
	count := 1
	b.WriteRune(runes[0])
	for i := 1; i < len(runes); i++ {
		if runes[i] == runes[i-1] {
			count++
		} else {
			count = 1
		}
		if count <= maxRepeats {
			b.WriteRune(runes[i])
		}
	}
	return b.String()
}

// --- SERVER CONTROL ---

func StartServer(modelName string, port int, cpuCore int) {
	serverMu.Lock()
	defer serverMu.Unlock()

	if ServerActive {
		return
	}

	LogMsg(fmt.Sprintf("[yellow]Initializing CGO engine for %s...[-]", modelName))

	onnxPath, configPath := getModelPaths(modelName)
	if onnxPath == "" {
		LogMsg("[red]Failed to find model files[-]")
		return
	}

	absPiper, _ := filepath.Abs(PiperDir)
	espeakData := filepath.Join(absPiper, "espeak-ng-data-v1-gpl")

	synth, err := piper.NewSynthesizer(onnxPath, configPath, espeakData)
	if err != nil {
		LogMsg(fmt.Sprintf("[red]Piper Init Error: %v[-]", err))
		return
	}

	activeSynthMu.Lock()
	if activeSynth != nil {
		activeSynth.Close()
	}
	activeSynth = synth
	activeSynthMu.Unlock()

	sampleRate := getSampleRate(configPath)
	ActiveModel = modelName
	ServerPort = port

	mux := http.NewServeMux()
	mux.HandleFunc("/tts", ttsHandler(sampleRate))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tts" {
			http.NotFound(w, r)
		}
	})

	serverSrv = &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}

	go func() {
		LogMsg(fmt.Sprintf("[green]CGO Server started on :%d using %s[-]", port, modelName))
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
	// We don't lock here because Start/Stop have their own locks
	// and we want to allow the UI update loop to read the state in between if needed
	if ServerActive {
		StopServer()
	} else {
		StartServer(model, port, cpuCore)
	}
}

// --- HTTP HANDLER ---

func ttsHandler(sampleRate int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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
		lengthScale := 1.0 / userSpeed
		lengthScale = math.Max(0.1, math.Min(lengthScale, 5.0))

		LogMsg(fmt.Sprintf("CGO Processing %d chars | Speed: %.1fx", len(text), userSpeed))

		activeSynthMu.Lock()
		synth := activeSynth
		activeSynthMu.Unlock()

		if synth == nil {
			http.Error(w, "Engine not initialized", 500)
			return
		}

		opts := synth.DefaultOptions()
		opts.LengthScale = float32(lengthScale)

		var audioData []byte
		err := synth.Synthesize(text, &opts, func(samples []float32, rate int) bool {
			// Convert float32 samples to int16 PCM
			for _, s := range samples {
				val := int16(s * 32767.0)
				audioData = append(audioData, byte(val&0xff), byte(val>>8))
			}
			return true
		})

		if err != nil {
			LogMsg(fmt.Sprintf("[red]Synthesize Error: %v[-]", err))
			http.Error(w, "Generation Failed", 500)
			return
		}

		w.Header().Set("Content-Type", "audio/wav")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		writeWavHeader(w, sampleRate, len(audioData))
		w.Write(audioData)
	}
}

// --- FILE HELPERS ---

func getModelPaths(key string) (string, string) {
	ConfigMu.RLock() // Use Registry via config lock or assume registry is immutable after load
	info, ok := Registry[key]
	ConfigMu.RUnlock()
	
	if !ok { return "", "" }
	var onnx, conf string
	for f := range info.Files {
		if strings.HasSuffix(f, ".onnx") { onnx = f }
		if strings.HasSuffix(f, ".onnx.json") { conf = f }
	}
	
	localOnnx := filepath.Join(ModelDir, filepath.Base(onnx))
	localConf := filepath.Join(ModelDir, filepath.Base(conf))
	os.MkdirAll(ModelDir, 0755)
	
	downloadIfNeeded(localOnnx, DownloadBase+onnx)
	downloadIfNeeded(localConf, DownloadBase+conf)
	return localOnnx, localConf
}

func downloadIfNeeded(path, url string) {
	if _, err := os.Stat(path); err == nil { return }
	LogMsg(fmt.Sprintf("Downloading %s...", filepath.Base(path)))
	resp, _ := http.Get(url)
	defer resp.Body.Close()
	out, _ := os.Create(path)
	io.Copy(out, resp.Body)
}

func getSampleRate(path string) int {
	f, _ := os.ReadFile(path)
	var d struct { Audio struct { SampleRate int `json:"sample_rate"` } `json:"audio"` }
	json.Unmarshal(f, &d)
	if d.Audio.SampleRate == 0 { return 22050 }
	return d.Audio.SampleRate
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