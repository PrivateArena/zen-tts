package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// --- DATA STRUCTURES ---

type EngineConfig struct {
	Model      string  `json:"model"`
	Voice      string  `json:"voice,omitempty"`
	Speed      float64 `json:"speed,omitempty"`
	ModelPath  string  `json:"model_path,omitempty"`
	ConfigPath string  `json:"config_path,omitempty"`
	Variation  float64 `json:"variation,omitempty"`
}

type ReplacementRule struct {
	Pattern     string `json:"pattern"`
	Replacement string `json:"replacement"`
	IsRegex     bool   `json:"is_regex"`
}

type Config struct {
	LastModel    string                  `json:"last_model"`
	Port         int                     `json:"port"`
	ActiveEngine string                  `json:"active_engine"`
	Engines      map[string]EngineConfig `json:"engines"`
	Replacements []ReplacementRule       `json:"replacements,omitempty"`
}

type VoiceRegistryEntry struct {
	Key      string `json:"key"`
	Language struct {
		Code        string `json:"code"`
		NameEnglish string `json:"name_english"`
	} `json:"language"`
	Quality string                 `json:"quality"`
	Files   map[string]interface{} `json:"files"`
}

type VoiceRegistry map[string]VoiceRegistryEntry

type TTSRequest struct {
	Text      string   `json:"text"`
	Voice     string   `json:"voice"`
	Speed     float64  `json:"speed"`
	Play      bool     `json:"play"`
	Stream    bool     `json:"stream"`
	Engine    string   `json:"engine"`
	Seed      *int64   `json:"seed,omitempty"`
	Variation *float64 `json:"variation,omitempty"`
}

// --- GLOBAL STATE ---

var (
	CurrentConfig Config
	Registry      VoiceRegistry
	PlainReplacer *strings.Replacer
	RegexRules    []RegexRule
	RawRules      []string
	ConfigMu      sync.RWMutex
)

type RegexRule struct {
	Re          *regexp.Regexp
	Replacement string
	Guard       string // Lowercase token that must exist in text to trigger regex
}

// --- METHODS ---

func LoadConfig() {
	// Defaults
	CurrentConfig = Config{
		LastModel:    "en_US-amy-low",
		Port:         5055,
		ActiveEngine: "piper",
		Engines: map[string]EngineConfig{
			"piper": {
				Model: "en_US-amy-low",
				Speed: 1.0,
			},
			"kokoro": {
				Model: "kokoro-v1.0",
				Voice: "af_bella",
				Speed: 1.0,
			},
			"kitten": {
				Model: "kitten-tts-mini",
				Voice: "luna",
				Speed: 1.0,
			},
			"inflect": {
				Model:     "inflect-micro-v2",
				Voice:     "default",
				Speed:     1.0,
				Variation: 0.667,
			},
		},
	}

	file, err := os.ReadFile(ConfigFile)
	if err == nil {
		json.Unmarshal(file, &CurrentConfig)
	}

	// Ensure the Engines map exists and is populated
	if CurrentConfig.Engines == nil {
		CurrentConfig.Engines = make(map[string]EngineConfig)
	}
	for _, engineType := range []string{"piper", "kokoro", "kitten", "inflect"} {
		if _, ok := CurrentConfig.Engines[engineType]; !ok {
			switch engineType {
			case "piper":
				CurrentConfig.Engines[engineType] = EngineConfig{Model: "en_US-amy-low", Speed: 1.0}
			case "kokoro":
				CurrentConfig.Engines[engineType] = EngineConfig{Model: "kokoro-v1.0", Voice: "af_bella", Speed: 1.0}
			case "kitten":
				CurrentConfig.Engines[engineType] = EngineConfig{Model: "kitten-tts-mini", Voice: "luna", Speed: 1.0}
			case "inflect":
				CurrentConfig.Engines[engineType] = EngineConfig{Model: "inflect-micro-v2", Voice: "default", Speed: 1.0, Variation: 0.667}
			}
		}
	}

	// Make sure active engine is valid
	if CurrentConfig.ActiveEngine == "" {
		CurrentConfig.ActiveEngine = "piper"
	}

	LoadReplacements()
}

func SaveConfig() {
	ConfigMu.Lock()
	defer ConfigMu.Unlock()

	data, _ := json.MarshalIndent(CurrentConfig, "", "  ")
	os.WriteFile(ConfigFile, data, 0644)
}

func LoadReplacements() {
	ConfigMu.Lock()
	defer ConfigMu.Unlock()

	var plainPairs []string
	var regexes []RegexRule
	var rawLines []string

	// 1. Read replacements.txt if exists
	data, err := os.ReadFile(ReplacementsFile)
	if err == nil {
		lines := strings.Split(string(data), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			rawLines = append(rawLines, line)

			parts := strings.Split(line, "::")
			if len(parts) < 2 {
				continue
			}

			pattern := parts[0]
			var guard, replacement string

			if len(parts) == 3 {
				guard = strings.ToLower(parts[1])
				replacement = strings.ReplaceAll(parts[2], "\\n", "\n")
			} else {
				replacement = strings.ReplaceAll(parts[1], "\\n", "\n")
			}

			if strings.HasPrefix(pattern, "re:") {
				cleanPat := strings.TrimPrefix(pattern, "re:")
				re, err := regexp.Compile("(?i)" + cleanPat)
				if err == nil {
					regexes = append(regexes, RegexRule{Re: re, Replacement: replacement, Guard: guard})
				}
			} else if strings.HasPrefix(pattern, "ci:") {
				cleanPat := strings.TrimPrefix(pattern, "ci:")
				re, err := regexp.Compile("(?i)" + regexp.QuoteMeta(cleanPat))
				if err == nil {
					if guard == "" {
						guard = strings.ToLower(cleanPat)
					}
					regexes = append(regexes, RegexRule{Re: re, Replacement: replacement, Guard: guard})
				}
			} else {
				plainPairs = append(plainPairs, pattern, replacement)
			}
		}
	} else {
		// Create empty replacements.txt if not exists
		os.WriteFile(ReplacementsFile, []byte(""), 0644)
	}

	// 2. Read config.json replacements
	for _, rule := range CurrentConfig.Replacements {
		if rule.IsRegex {
			re, err := regexp.Compile("(?i)" + rule.Pattern)
			if err == nil {
				regexes = append(regexes, RegexRule{Re: re, Replacement: rule.Replacement})
			}
		} else {
			plainPairs = append(plainPairs, rule.Pattern, rule.Replacement)
		}
		// Form a raw rule representation for the UI table
		rawRuleLine := rule.Pattern + "::" + rule.Replacement
		if rule.IsRegex {
			rawRuleLine = "re:" + rawRuleLine
		}
		rawLines = append(rawLines, rawRuleLine)
	}

	PlainReplacer = strings.NewReplacer(plainPairs...)
	RegexRules = regexes
	RawRules = rawLines
}

func LoadRegistry() {
	file, err := os.ReadFile(VoicesFile)
	if err != nil {
		log.Fatalf("❌ Error: %s not found. Please download it.", VoicesFile)
	}
	if err := json.Unmarshal(file, &Registry); err != nil {
		log.Fatalf("❌ Error parsing %s: %v", VoicesFile, err)
	}

	// Dynamic injection of Kokoro-82M into the registry
	kokoroEntry := VoiceRegistryEntry{}
	kokoroEntry.Key = "kokoro-v1.0"
	kokoroEntry.Language.Code = "en_US"
	kokoroEntry.Language.NameEnglish = "English (Kokoro Realism)"
	kokoroEntry.Quality = "high"
	kokoroEntry.Files = map[string]interface{}{
		"kokoro-v1.0.onnx": nil,
		"config.json":      nil,
	}
	Registry["kokoro-v1.0"] = kokoroEntry

	// Dynamic injection of KittenTTS models into the registry using a clean loop
	for _, size := range []struct {
		key     string
		name    string
		quality string
	}{
		{"kitten-tts-mini", "English (KittenTTS Mini)", "high"},
		{"kitten-tts-micro", "English (KittenTTS Micro)", "medium"},
		{"kitten-tts-nano", "English (KittenTTS Nano)", "low"},
	} {
		entry := VoiceRegistryEntry{
			Key:     size.key,
			Quality: size.quality,
		}
		entry.Language.Code = "en_US"
		entry.Language.NameEnglish = size.name
		entry.Files = map[string]interface{}{
			size.key + ".onnx": nil,
			"config.json":      nil,
		}
		Registry[size.key] = entry
	}
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

	var localOnnx, localConf string
	if strings.HasPrefix(key, "inflect") {
		localDir := filepath.Join(ModelDir, "inflect", key)
		localOnnx = filepath.Join(localDir, "decode.onnx")
		localConf = filepath.Join(localDir, "config.json")
		os.MkdirAll(localDir, 0755)
	} else if strings.HasPrefix(key, "kokoro") {
		localOnnx = filepath.Join(ModelDir, "kokoro", "kokoro-v1.0.onnx")
		localConf = filepath.Join(ModelDir, "kokoro", "kokoro-v1.0.json")
		os.MkdirAll(filepath.Join(ModelDir, "kokoro"), 0755)
	} else if strings.HasPrefix(key, "kitten") {
		localOnnx = filepath.Join(ModelDir, "kitten", key+".onnx")
		localConf = filepath.Join(ModelDir, "kitten", key+".json")
		os.MkdirAll(filepath.Join(ModelDir, "kitten"), 0755)
	} else {
		localOnnx = filepath.Join(ModelDir, "piper", filepath.Base(onnx))
		localConf = filepath.Join(ModelDir, "piper", filepath.Base(conf))
		os.MkdirAll(filepath.Join(ModelDir, "piper"), 0755)
	}

	if strings.HasPrefix(key, "inflect") {
		localDir := filepath.Join(ModelDir, "inflect", key)
		os.MkdirAll(localDir, 0755)
		downloadIfNeeded(filepath.Join(localDir, "duration.onnx"), InflectMicroOnnxDurationURL)
		downloadIfNeeded(filepath.Join(localDir, "decode.onnx"), InflectMicroOnnxDecodeURL)
		downloadIfNeeded(filepath.Join(localDir, "config.json"), InflectMicroConfigURL)
		return localDir, filepath.Join(localDir, "config.json")
	} else if strings.HasPrefix(key, "kokoro") {
		// Auto-download Kokoro-82M onnx community model and config
		downloadIfNeeded(localOnnx, KokoroOnnxURL)
		downloadIfNeeded(localConf, KokoroConfURL)

		// Create voices directory and download default voice
		voicesDir := filepath.Join(ModelDir, "kokoro", "voices")
		os.MkdirAll(voicesDir, 0755)
		downloadIfNeeded(filepath.Join(voicesDir, "af_bella.bin"), KokoroBellaURL)
	} else if strings.HasPrefix(key, "kitten") {
		// Auto-download KittenTTS onnx and config
		onnxURL := KittenMiniOnnxURL
		confURL := KittenMiniConfURL
		if key == "kitten-tts-micro" {
			onnxURL = KittenMicroOnnxURL
			confURL = KittenMicroConfURL
		} else if key == "kitten-tts-nano" {
			onnxURL = KittenNanoOnnxURL
			confURL = KittenNanoConfURL
		}
		downloadIfNeeded(localOnnx, onnxURL)
		downloadIfNeeded(localConf, confURL)
		downloadIfNeeded(filepath.Join(ModelDir, "kitten", "voices.npz"), KittenVoicesURL)
	} else {
		// Piper-specific download URLs
		downloadIfNeeded(localOnnx, PiperDownloadBase+onnx)
		downloadIfNeeded(localConf, PiperDownloadBase+conf)
	}

	return localOnnx, localConf
}

func downloadIfNeeded(path, url string) {
	if fi, err := os.Stat(path); err == nil {
		if fi.Size() > 200 {
			return
		}
		os.Remove(path)
	}
	LogMsg(fmt.Sprintf("Downloading %s...", filepath.Base(path)))
	resp, err := http.Get(url)
	if err != nil {
		LogMsg(fmt.Sprintf("[red]Download error: %v[-]", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		LogMsg(fmt.Sprintf("[red]Download failed for %s: HTTP %d %s[-]", filepath.Base(path), resp.StatusCode, resp.Status))
		return
	}

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
