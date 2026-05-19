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

	"github.com/ikawaha/kagome-dict/ipa"
	"github.com/ikawaha/kagome/v2/tokenizer"
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

// Phonemize converts raw text into IPA phonemes using the local espeak-ng/piper_phonemize subprocess for a specific language
func Phonemize(text string, lang string) ([]string, error) {
	if lang == "" {
		lang = "en-us"
	}
	absPiper, _ := filepath.Abs("./piper")
	cmdPath := filepath.Join(absPiper, "piper_phonemize")
	espeakData := filepath.Join(absPiper, "espeak-ng-data-v1-gpl")

	cmd := exec.Command(cmdPath, "-l", lang, "--espeak_data", espeakData)
	cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+absPiper)

	var stdin bytes.Buffer
	stdin.WriteString(text + "\n")
	cmd.Stdin = &stdin

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err := cmd.Run()
	stdoutStr := stdout.String()
	stderrStr := stderr.String()

	// First try parsing stdout as a whole
	var result struct {
		Phonemes []string `json:"phonemes"`
	}
	if json.Unmarshal([]byte(stdoutStr), &result) == nil && len(result.Phonemes) > 0 {
		return result.Phonemes, nil
	}

	// If stdout failed or was empty, search both stdout and stderr line-by-line for JSON block
	combined := stdoutStr + "\n" + stderrStr
	lines := strings.Split(combined, "\n")
	for _, line := range lines {
		idx := strings.Index(line, `{"phoneme_ids"`)
		if idx == -1 {
			idx = strings.Index(line, `{"phonemes"`)
		}
		if idx == -1 {
			idx = strings.Index(line, "{")
		}
		if idx != -1 {
			jsonStr := line[idx:]
			var res struct {
				Phonemes []string `json:"phonemes"`
			}
			if json.Unmarshal([]byte(jsonStr), &res) == nil && len(res.Phonemes) > 0 {
				return res.Phonemes, nil
			}
		}
	}

	// If everything failed, report error
	if err != nil {
		return nil, fmt.Errorf("failed to run phonemize: %v, stderr: %s", err, stderrStr)
	}
	return nil, fmt.Errorf("could not find valid phonemes JSON in outputs (stdout len: %d, stderr len: %d)", len(stdoutStr), len(stderrStr))
}

// GetLanguageCode extracts the piper_phonemize language code based on Kokoro voice name prefixes
func GetLanguageCode(voice string) string {
	voice = strings.ToLower(voice)
	if strings.HasPrefix(voice, "bf_") || strings.HasPrefix(voice, "bm_") {
		return "en-gb"
	}
	if strings.HasPrefix(voice, "zf_") || strings.HasPrefix(voice, "zm_") {
		return "cmn"
	}
	if strings.HasPrefix(voice, "jf_") || strings.HasPrefix(voice, "jm_") {
		return "ja"
	}
	if strings.HasPrefix(voice, "ef_") || strings.HasPrefix(voice, "em_") {
		return "es"
	}
	if strings.HasPrefix(voice, "ff_") {
		return "fr"
	}
	if strings.HasPrefix(voice, "if_") || strings.HasPrefix(voice, "im_") {
		return "it"
	}
	if strings.HasPrefix(voice, "pf_") || strings.HasPrefix(voice, "pm_") {
		return "pt-br"
	}
	if strings.HasPrefix(voice, "hf_") || strings.HasPrefix(voice, "hm_") {
		return "hi"
	}
	return "en-us"
}

// Tokenize converts IPA phonemes or fallback characters into an ONNX-compatible ID sequence for English (US)
func Tokenize(text string) []int64 {
	return TokenizeWithVoice(text, "en-us")
}

var (
	jpTokenizer     *tokenizer.Tokenizer
	jpTokenizerOnce sync.Once
)

func getJpTokenizer() *tokenizer.Tokenizer {
	jpTokenizerOnce.Do(func() {
		t, err := tokenizer.New(ipa.Dict(), tokenizer.OmitBosEos())
		if err == nil {
			jpTokenizer = t
		}
	})
	return jpTokenizer
}

func ConvertKanjiToHiragana(text string) string {
	t := getJpTokenizer()
	if t == nil {
		return text
	}
	tokens := t.Tokenize(text)
	var result []string
	for _, token := range tokens {
		reading, ok := token.Reading()
		if ok {
			result = append(result, katakanaToHiragana(reading))
		} else {
			result = append(result, token.Surface)
		}
	}
	return strings.Join(result, "")
}

func katakanaToHiragana(s string) string {
	var sb strings.Builder
	for _, r := range s {
		if r >= 0x30A1 && r <= 0x30F6 {
			sb.WriteRune(r - 0x60)
		} else {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

// TokenizeWithVoice converts IPA phonemes into an ONNX ID sequence using the language code mapped to a specific voice
func TokenizeWithVoice(text string, voice string) []int64 {
	var tokens []int64
	tokens = append(tokens, 0) // Start token

	lang := GetLanguageCode(voice)
	if lang == "ja" {
		text = ConvertKanjiToHiragana(text)
	}

	phonemes, err := Phonemize(text, lang)
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
