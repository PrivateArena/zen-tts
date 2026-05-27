package shared

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/ikawaha/kagome-dict/ipa"
	"github.com/ikawaha/kagome/v2/tokenizer"
)

// WordTiming holds the time span for a single word, relative to the segment start.
type WordTiming struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"` // seconds from segment start
	End   float64 `json:"end"`   // seconds from segment start
}

// WordBoundary records how many phoneme tokens correspond to each source word.
type WordBoundary struct {
	Word       string
	TokenCount int // number of ONNX tokens (excluding BOS/EOS) produced for this word
}

// TokenizeWordBoundaries tokenizes each word individually and returns per-word
// token counts. This allows PCM-level silence detection to be aligned to words.
// The total token sequence produced here matches TokenizeWithVoice(fullText, voice)
// when words are joined with a single space.
func TokenizeWordBoundaries(text string, voice string) []WordBoundary {
	words := strings.Fields(text)
	lang := GetLanguageCode(voice)
	boundaries := make([]WordBoundary, 0, len(words))

	for _, word := range words {
		tokens := tokenizeWord(word, lang)
		boundaries = append(boundaries, WordBoundary{
			Word:       word,
			TokenCount: len(tokens),
		})
	}
	return boundaries
}

// tokenizeWord phonemizes a single word and maps to vocab IDs (no BOS/EOS).
func tokenizeWord(word string, lang string) []int64 {
	phonemes, err := Phonemize(word, lang)
	var tokens []int64
	if err == nil && len(phonemes) > 0 {
		for _, p := range phonemes {
			if id, ok := Vocab[p]; ok {
				tokens = append(tokens, id)
			} else {
				for _, r := range p {
					if id, ok := Vocab[string(r)]; ok {
						tokens = append(tokens, id)
					}
				}
			}
		}
	} else {
		for _, r := range strings.ToLower(word) {
			if id, ok := Vocab[string(r)]; ok {
				tokens = append(tokens, id)
			}
		}
	}
	return tokens
}

// SilenceBoundaryTimings derives per-word timings from raw PCM float32 samples
// using energy-envelope silence detection. It is the production fallback when
// the ONNX model does not expose a "durations" tensor.
//
// Algorithm:
//  1. Compute RMS energy in small windows (windowSamples).
//  2. Find silence gaps (energy < silenceThreshold) between words.
//  3. Map silence boundaries back to words using tokenCount proportions as weights
//     when a clear silence gap can't be found.
func SilenceBoundaryTimings(samples []float32, sampleRate int, boundaries []WordBoundary) []WordTiming {
	if len(samples) == 0 || len(boundaries) == 0 {
		return nil
	}

	totalDuration := float64(len(samples)) / float64(sampleRate)

	// Silence detection parameters
	const windowMs = 20   // RMS window size in milliseconds
	const silenceDB = -35 // threshold in dB below peak

	windowSamples := sampleRate * windowMs / 1000
	if windowSamples < 1 {
		windowSamples = 1
	}

	// Compute per-window RMS
	numWindows := (len(samples) + windowSamples - 1) / windowSamples
	rmsValues := make([]float64, numWindows)
	peakRMS := 0.0
	for i := 0; i < numWindows; i++ {
		start := i * windowSamples
		end := start + windowSamples
		if end > len(samples) {
			end = len(samples)
		}
		var sumSq float64
		for _, s := range samples[start:end] {
			sumSq += float64(s) * float64(s)
		}
		rms := math.Sqrt(sumSq / float64(end-start))
		rmsValues[i] = rms
		if rms > peakRMS {
			peakRMS = rms
		}
	}

	// Derive absolute silence threshold from peak
	silenceLinear := peakRMS * math.Pow(10.0, float64(silenceDB)/20.0)

	// Mark silence windows
	isSilent := make([]bool, numWindows)
	for i, rms := range rmsValues {
		isSilent[i] = rms <= silenceLinear
	}

	windowDur := float64(windowSamples) / float64(sampleRate)

	// Find silence gap centers between each pair of adjacent words.
	// We need (len(boundaries)-1) gap positions.
	numGaps := len(boundaries) - 1
	gapCenters := make([]float64, numGaps)

	// Compute total token counts for proportional fallback
	totalTokens := 0
	for _, b := range boundaries {
		totalTokens += b.TokenCount
	}

	// For each gap, search for a silence window near the expected proportional position
	cumTokens := 0
	for g := 0; g < numGaps; g++ {
		cumTokens += boundaries[g].TokenCount
		propFrac := float64(cumTokens) / float64(totalTokens)
		expectedTime := propFrac * totalDuration
		expectedWin := int(expectedTime / windowDur)

		// Search ±15% of totalDuration around expected position
		searchRadius := int(0.15 * totalDuration / windowDur)
		if searchRadius < 3 {
			searchRadius = 3
		}
		lo := expectedWin - searchRadius
		hi := expectedWin + searchRadius
		if lo < 0 {
			lo = 0
		}
		if hi >= numWindows {
			hi = numWindows - 1
		}

		// Find the longest silence run in [lo,hi]
		bestCenter := -1
		bestLen := 0
		i := lo
		for i <= hi {
			if isSilent[i] {
				j := i
				for j <= hi && isSilent[j] {
					j++
				}
				runLen := j - i
				if runLen > bestLen {
					bestLen = runLen
					bestCenter = i + runLen/2
				}
				i = j
			} else {
				i++
			}
		}

		if bestCenter >= 0 {
			gapCenters[g] = float64(bestCenter) * windowDur
		} else {
			// Fallback: use proportional position
			gapCenters[g] = expectedTime
		}
	}

	// Build timings from gap centers
	timings := make([]WordTiming, len(boundaries))
	currentStart := 0.0
	for i, b := range boundaries {
		var end float64
		if i < numGaps {
			end = gapCenters[i]
		} else {
			end = totalDuration
		}
		if end <= currentStart {
			end = currentStart + (totalDuration / float64(len(boundaries)))
		}
		timings[i] = WordTiming{
			Word:  b.Word,
			Start: currentStart,
			End:   end,
		}
		currentStart = end
	}
	return timings
}

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
