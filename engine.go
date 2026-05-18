package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"zen-tts/internal/kitten"
	"zen-tts/internal/kokoro"
	"zen-tts/internal/piper"
)

var (
	ServerActive  bool
	ServerPort    int
	ActiveModel   string
	serverSrv     *http.Server
	serverMu      sync.Mutex
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

// --- FILE HELPERS ---

func getModelPaths(key string) (string, string) {
	ConfigMu.RLock()
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
		if strings.HasSuffix(f, ".json") {
			conf = f
		}
	}

	localOnnx := filepath.Join(ModelDir, filepath.Base(onnx))
	localConf := filepath.Join(ModelDir, filepath.Base(conf))
	if strings.HasPrefix(key, "kokoro") {
		localOnnx = filepath.Join(ModelDir, "kokoro-v0.19.onnx")
		localConf = filepath.Join(ModelDir, "kokoro-v0.19.json")
	} else if strings.HasPrefix(key, "kitten") {
		localOnnx = filepath.Join(ModelDir, key+".onnx")
		localConf = filepath.Join(ModelDir, key+".json")
	}
	os.MkdirAll(ModelDir, 0755)

	if strings.HasPrefix(key, "kokoro") {
		// Auto-download Kokoro-82M onnx community model and config
		onnxURL := "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"
		confURL := "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/config.json"
		downloadIfNeeded(localOnnx, onnxURL)
		downloadIfNeeded(localConf, confURL)

		// Create voices directory and download default voice
		voicesDir := filepath.Join(ModelDir, "voices")
		os.MkdirAll(voicesDir, 0755)
		downloadIfNeeded(filepath.Join(voicesDir, "af_bella.bin"), "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_bella.bin")
		downloadIfNeeded(filepath.Join(voicesDir, "af_jasper.bin"), "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_jasper.bin")
	} else if strings.HasPrefix(key, "kitten") {
		// Auto-download KittenTTS onnx and config
		onnxURL := "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/kitten_tts_mini_v0_8.onnx"
		confURL := "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/config.json"
		if key == "kitten-tts-micro" {
			onnxURL = "https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/kitten_tts_micro_v0_8.onnx"
			confURL = "https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/config.json"
		} else if key == "kitten-tts-nano" {
			onnxURL = "https://huggingface.co/KittenML/kitten-tts-nano-0.8/resolve/main/kitten_tts_nano_v0_8.onnx"
			confURL = "https://huggingface.co/KittenML/kitten-tts-nano-0.8/resolve/main/config.json"
		}
		downloadIfNeeded(localOnnx, onnxURL)
		downloadIfNeeded(localConf, confURL)
		downloadIfNeeded(filepath.Join(ModelDir, "voices.npz"), "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/voices.npz")
	} else {
		// Piper-specific download URLs
		downloadIfNeeded(localOnnx, DownloadBase+onnx)
		downloadIfNeeded(localConf, DownloadBase+conf)
	}

	return localOnnx, localConf
}

func downloadIfNeeded(path, url string) {
	if _, err := os.Stat(path); err == nil {
		return
	}
	LogMsg(fmt.Sprintf("Downloading %s...", filepath.Base(path)))
	resp, err := http.Get(url)
	if err != nil {
		LogMsg(fmt.Sprintf("[red]Download error: %v[-]", err))
		return
	}
	defer resp.Body.Close()
	out, err := os.Create(path)
	if err != nil {
		LogMsg(fmt.Sprintf("[red]Error creating file %s: %v[-]", path, err))
		return
	}
	defer out.Close()
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
