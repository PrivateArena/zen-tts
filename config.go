package main

import (
	"encoding/json"
	"log"
	"os"
	"regexp"
	"sync"
)

const (
	VoicesFile   = "voices.json"
	ConfigFile   = "config.json"
	PiperDir     = "./piper"
	PiperBinary  = "./piper/piper"
	ModelDir     = "./models"
	DownloadBase = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
)

// --- DATA STRUCTURES ---

type ReplacementRule struct {
	Pattern     string `json:"pattern"`
	Replacement string `json:"replacement"`
	IsRegex     bool   `json:"is_regex"`
}

type Config struct {
	LastModel    string            `json:"last_model"`
	Port         int               `json:"port"`
	Replacements []ReplacementRule `json:"replacements"`
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
	RegexCache    map[string]*regexp.Regexp
	ConfigMu      sync.RWMutex
)

// --- METHODS ---

func LoadConfig() {
	RegexCache = make(map[string]*regexp.Regexp)
	
	// Defaults
	CurrentConfig = Config{
		Port:      5000,
		LastModel: "en_US-amy-low",
		Replacements: []ReplacementRule{
			{Pattern: "Kee{3,}", Replacement: "Key", IsRegex: true},
			{Pattern: "https?://\\S+", Replacement: "Link removed", IsRegex: true},
		},
	}

	file, err := os.ReadFile(ConfigFile)
	if err == nil {
		json.Unmarshal(file, &CurrentConfig)
	}

	CompileRegex()
}

func SaveConfig() {
	ConfigMu.Lock()
	defer ConfigMu.Unlock()
	
	data, _ := json.MarshalIndent(CurrentConfig, "", "  ")
	os.WriteFile(ConfigFile, data, 0644)
	
	// Recompile regex whenever config is saved/changed
	CompileRegex()
}

func CompileRegex() {
	// Compile regex safely
	for _, r := range CurrentConfig.Replacements {
		if r.IsRegex {
			re, err := regexp.Compile("(?i)" + r.Pattern)
			if err == nil {
				RegexCache[r.Pattern] = re
			}
		}
	}
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