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
	"regexp"
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

	// 1. High-Performance Plain Text Replacements (O(N) pass)
	if PlainReplacer != nil {
		text = PlainReplacer.Replace(text)
	}

	// 2. Regex Replacements (Only for complex patterns)
	lowerText := strings.ToLower(text)
	for _, rule := range RegexRules {
		if rule.Guard != "" && !strings.Contains(lowerText, rule.Guard) {
			continue
		}
		text = rule.Re.ReplaceAllString(text, rule.Replacement)
	}

	// 3. Advanced Phrasing & Prosodic Emphasis
	text = applyProsodicEmphasis(text)
	return heuristicPhrasing(text)
}

func applyProsodicEmphasis(text string) string {
	// Detects: word followed by multiple ! or ? (e.g., "Wait!!!")
	// 1. Injects a comma-pause BEFORE for "gravity"
	// 2. Uppercases the word for Piper's "energy" boost
	// 3. Normalizes punctuation to a consistent emphatic ending
	reShout := regexp.MustCompile(`(\b[a-zA-Z]{2,})([!]{2,}|[?]{2,})`)
	return reShout.ReplaceAllStringFunc(text, func(m string) string {
		parts := reShout.FindStringSubmatch(m)
		if len(parts) < 3 {
			return m
		}
		// A preceding comma forces a slight pause/intonation reset which makes the shout impactful
		return ", " + strings.ToUpper(parts[1]) + parts[2][:1] + "!"
	})
}

func heuristicPhrasing(text string) string {
	// A. Convert newlines to periods if no punctuation exists
	// This handles "list-like" bad writing where people skip dots at line ends
	lines := strings.Split(text, "\n")
	changed := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if len(trimmed) > 0 {
			lastChar := trimmed[len(trimmed)-1]
			if lastChar != '.' && lastChar != '!' && lastChar != '?' && lastChar != ',' && lastChar != ':' && lastChar != ';' {
				lines[i] = trimmed + "."
				changed = true
			}
		}
	}
	if changed {
		text = strings.Join(lines, " ")
	}

	// B. Force a "breath" every 18 words to prevent run-on sentences
	words := strings.Fields(text)
	if len(words) > 20 {
		var b strings.Builder
		wordsSincePunc := 0
		for _, w := range words {
			b.WriteString(w)
			b.WriteString(" ")
			wordsSincePunc++

			hasPunc := strings.ContainsAny(w, ".,!?;:")
			if hasPunc {
				wordsSincePunc = 0
			} else if wordsSincePunc > 18 {
				b.WriteString(", ") // Inject a comma "breath"
				wordsSincePunc = 0
			}
		}
		text = strings.TrimSpace(b.String())
	}

	return text
}

func squashRepeats(s string, maxRepeats int) string {
	if len(s) == 0 {
		return s
	}
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

	if !ok {
		return "", ""
	}
	var onnx, conf string
	for f := range info.Files {
		if strings.HasSuffix(f, ".onnx") {
			onnx = f
		}
		if strings.HasSuffix(f, ".onnx.json") {
			conf = f
		}
	}

	localOnnx := filepath.Join(ModelDir, filepath.Base(onnx))
	localConf := filepath.Join(ModelDir, filepath.Base(conf))
	os.MkdirAll(ModelDir, 0755)

	downloadIfNeeded(localOnnx, DownloadBase+onnx)
	downloadIfNeeded(localConf, DownloadBase+conf)
	return localOnnx, localConf
}

func downloadIfNeeded(path, url string) {
	if _, err := os.Stat(path); err == nil {
		return
	}
	LogMsg(fmt.Sprintf("Downloading %s...", filepath.Base(path)))
	resp, _ := http.Get(url)
	defer resp.Body.Close()
	out, _ := os.Create(path)
	io.Copy(out, resp.Body)
}

func getSampleRate(path string) int {
	f, _ := os.ReadFile(path)
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
