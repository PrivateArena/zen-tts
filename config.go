package main

import (
	"encoding/json"
	"log"
	"os"
	"regexp"
	"strings"
	"sync"
)

const (
	VoicesFile       = "voices.json"
	ConfigFile       = "config.json"
	ReplacementsFile = "replacements.txt"
	PiperDir         = "./piper"
	PiperBinary      = "./piper/piper"
	ModelDir         = "./models"
	DownloadBase     = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
)

// --- DATA STRUCTURES ---

type Config struct {
	LastModel string `json:"last_model"`
	Port      int    `json:"port"`
}

type VoiceRegistry map[string]struct {
	Key      string `json:"key"`
	Language struct {
		Code        string `json:"code"`
		NameEnglish string `json:"name_english"`
	} `json:"language"`
	Quality string                 `json:"quality"`
	Files   map[string]interface{} `json:"files"`
}

type TTSRequest struct {
	Text  string  `json:"text"`
	Speed float64 `json:"speed"`
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
		Port:      5000,
		LastModel: "en_US-amy-low",
	}

	file, err := os.ReadFile(ConfigFile)
	if err == nil {
		json.Unmarshal(file, &CurrentConfig)
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

	data, err := os.ReadFile(ReplacementsFile)
	if err != nil {
		// Create empty file if not exists
		os.WriteFile(ReplacementsFile, []byte(""), 0644)
		return
	}

	lines := strings.Split(string(data), "\n")
	var plainPairs []string
	var regexes []RegexRule
	var rawLines []string

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
				// For ci: rules, the guard is just the lowercased pattern itself if not provided
				if guard == "" {
					guard = strings.ToLower(cleanPat)
				}
				regexes = append(regexes, RegexRule{Re: re, Replacement: replacement, Guard: guard})
			}
		} else {
			plainPairs = append(plainPairs, pattern, replacement)
		}
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
}