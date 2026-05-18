package shared

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

// InitONNXRuntime initializes the shared ONNX Runtime engine.
func InitONNXRuntime() error {
	ortInitOnce.Do(func() {
		// Look for libonnxruntime.so inside piper directory first
		libPath := "./piper/libonnxruntime.so.1.24.2"
		if _, err := os.Stat(libPath); err != nil {
			libPath = "./piper/libonnxruntime.so"
		}
		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// Vocab is the official, 100% correct Kokoro & KittenTTS IPA phoneme vocabulary map.
var Vocab = map[string]int64{
	"$": 0, ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10,
	"\"": 11, "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "̃": 17, "ʣ": 18,
	"ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "ꭧ": 23, "A": 24, "I": 25, "O": 31,
	"Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44,
	"c": 45, "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53,
	"l": 54, "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61,
	"t": 62, "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69,
	"ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80,
	"ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92,
	"ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112,
	"ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120,
	"ɹ": 123, "ɾ": 125, "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131,
	"ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140,
	"χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157, "ː": 158,
	"ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177,
}

// Phonemize converts raw English text into IPA phonemes using the local espeak-ng/piper_phonemize subprocess
func Phonemize(text string) ([]string, error) {
	absPiper, _ := filepath.Abs("./piper")
	cmdPath := filepath.Join(absPiper, "piper_phonemize")
	espeakData := filepath.Join(absPiper, "espeak-ng-data-v1-gpl")

	cmd := exec.Command(cmdPath, "-l", "en-us", "--espeak_data", espeakData)
	cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+absPiper)

	var stdin bytes.Buffer
	stdin.WriteString(text + "\n")
	cmd.Stdin = &stdin

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run phonemize: %v, stderr: %s", err, stderr.String())
	}

	var result struct {
		Phonemes []string `json:"phonemes"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal phonemizer output: %v", err)
	}

	return result.Phonemes, nil
}

// Tokenize converts IPA phonemes or fallback characters into an ONNX-compatible ID sequence
func Tokenize(text string) []int64 {
	var tokens []int64
	tokens = append(tokens, 0) // Start token

	// Generate IPA phonemes using our robust subprocess phonemizer
	phonemes, err := Phonemize(text)
	if err == nil && len(phonemes) > 0 {
		for _, p := range phonemes {
			if id, ok := Vocab[p]; ok {
				tokens = append(tokens, id)
			} else {
				// Map individual runes if it's a combined IPA character
				for _, r := range p {
					if id, ok := Vocab[string(r)]; ok {
						tokens = append(tokens, id)
					}
				}
			}
		}
	} else {
		// Fallback to character mapping if phonemizer fails
		text = strings.ToLower(text)
		for _, r := range text {
			symbol := string(r)
			if id, ok := Vocab[symbol]; ok {
				tokens = append(tokens, id)
			}
		}
	}

	tokens = append(tokens, 0) // End token
	return tokens
}
