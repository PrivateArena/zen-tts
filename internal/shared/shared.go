package shared

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
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
//
// Note that the sum of the counts here is NOT the length of
// TokenizeWithVoice(fullText, voice): per-word phonemization drops the inter-word
// space phonemes, so this under-counts by roughly one token per gap. Use
// TokenStream.WordBoundaries when the exact synthesized token stream matters, and
// keep this only as the fallback for text whose space grouping will not line up.
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
// using energy-envelope silence detection. It is the production fallback when the
// ONNX model does not expose a "durations" tensor; its error is bounded by the pause
// structure of the utterance, not by the frame size.
//
// Algorithm:
//  1. RMS energy in 10 ms windows.
//  2. Silence threshold relative to the 90th percentile window, not the peak: one
//     loud word must not be able to hide every quiet one behind it.
//  3. Assign every word boundary to one of the interior silence runs with a single
//     monotone dynamic program (cost = distance from the token-count prediction,
//     discounted by how long the pause is). Searching each gap independently is what
//     used to produce boundaries that jumped backwards in time, and the equal-slice
//     patch applied there invented word boundaries that never happened.
//  4. Clamp the first word to the end of the leading silence and the last word to
//     the start of the trailing silence.
func SilenceBoundaryTimings(samples []float32, sampleRate int, boundaries []WordBoundary) []WordTiming {
	if len(samples) == 0 || len(boundaries) == 0 || sampleRate <= 0 {
		return nil
	}

	totalDuration := float64(len(samples)) / float64(sampleRate)

	const silenceDB = -35.0 // threshold in dB below the p90 speech window

	windowSamples := sampleRate * VADWindowMS / 1000
	if windowSamples < 1 {
		windowSamples = 1
	}
	numWindows := (len(samples) + windowSamples - 1) / windowSamples
	if numWindows == 0 {
		return nil
	}
	windowDur := float64(windowSamples) / float64(sampleRate)

	rmsValues := make([]float64, numWindows)
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
		rmsValues[i] = math.Sqrt(sumSq / float64(end-start))
	}

	speechLevel := p90(rmsValues)
	isSilent := make([]bool, numWindows)
	if speechLevel <= 0 {
		// Pure digital silence: nothing to align to, spread the words proportionally.
		return proportionalTimings(boundaries, 0, totalDuration)
	}
	silenceLinear := speechLevel * math.Pow(10.0, silenceDB/20.0)
	for i, rms := range rmsValues {
		isSilent[i] = rms <= silenceLinear
	}

	// Measured speech span: the leading and trailing silence is not part of any word.
	firstSpeech, lastSpeech := -1, -1
	for i := range isSilent {
		if !isSilent[i] {
			if firstSpeech < 0 {
				firstSpeech = i
			}
			lastSpeech = i
		}
	}
	speechStart, speechEnd := 0.0, totalDuration
	if firstSpeech >= 0 {
		speechStart = float64(firstSpeech) * windowDur
		speechEnd = math.Min(float64(lastSpeech+1)*windowDur, totalDuration)
	}
	if speechEnd <= speechStart {
		speechStart, speechEnd = 0, totalDuration
	}

	if len(boundaries) == 1 {
		return []WordTiming{{Word: boundaries[0].Word, Start: speechStart, End: speechEnd}}
	}

	// Interior silence runs are the measured boundary candidates.
	type pause struct {
		pos  float64
		secs float64
	}
	var pauses []pause
	maxPause := 0.0
	if firstSpeech >= 0 {
		for i := firstSpeech; i <= lastSpeech; {
			if !isSilent[i] {
				i++
				continue
			}
			j := i
			for j <= lastSpeech && isSilent[j] {
				j++
			}
			d := float64(j-i) * windowDur
			pauses = append(pauses, pause{pos: float64(i+j) / 2 * windowDur, secs: d})
			if d > maxPause {
				maxPause = d
			}
			i = j
		}
	}

	numGaps := len(boundaries) - 1

	// Token-count prior over the measured speech span.
	totalTokens := 0
	for _, b := range boundaries {
		if b.TokenCount > 0 {
			totalTokens += b.TokenCount
		}
	}
	if totalTokens <= 0 {
		totalTokens = len(boundaries)
	}
	predicted := make([]float64, numGaps)
	cumTokens := 0
	for g := 0; g < numGaps; g++ {
		if boundaries[g].TokenCount > 0 {
			cumTokens += boundaries[g].TokenCount
		} else {
			cumTokens++
		}
		predicted[g] = speechStart + float64(cumTokens)/float64(totalTokens)*(speechEnd-speechStart)
		if g > 0 && predicted[g] <= predicted[g-1] {
			predicted[g] = predicted[g-1] + windowDur // keep the prior strictly increasing
		}
	}

	// Candidate pool: every measured pause plus one "no pause found" candidate per
	// boundary, sorted by time, so that an increasing candidate index is an increasing
	// time and the alignment below is monotone by construction.
	cands := make([]alignCandidate, 0, len(pauses)+numGaps)
	for _, p := range pauses {
		discount := 1.0
		if maxPause > 0 {
			discount = 1.0 - math.Min(1.0, p.secs/maxPause)
		}
		cands = append(cands, alignCandidate{pos: p.pos, cost: discount, owner: -1})
	}
	for g := 0; g < numGaps; g++ {
		cands = append(cands, alignCandidate{pos: predicted[g], cost: 1.0, owner: g})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].pos < cands[j].pos })

	ownIdx := make([]int, numGaps)
	for i := range ownIdx {
		ownIdx[i] = -1
	}
	for i, c := range cands {
		if c.owner >= 0 {
			ownIdx[c.owner] = i
		}
	}

	meanGap := (speechEnd - speechStart) / float64(numGaps+1)
	if meanGap <= 0 {
		meanGap = totalDuration / float64(numGaps+1)
	}
	// Sliding window: a boundary never wanders more than two average gaps from its
	// prediction, which also keeps the dynamic program linear in the word count.
	searchWin := 2 * meanGap
	if searchWin < 10*windowDur {
		searchWin = 10 * windowDur
	}

	stages := make([]alignStage, numGaps)
	for g := 0; g < numGaps; g++ {
		idx := windowCandidates(cands, predicted[g], searchWin, ownIdx[g])
		stage := alignStage{
			idx:   idx,
			cost:  make([]float64, len(idx)),
			from:  make([]int, len(idx)),
			best:  math.Inf(1),
			bestK: -1,
		}
		for k, j := range idx {
			local := math.Abs(cands[j].pos-predicted[g])/meanGap + cands[j].cost
			if g == 0 {
				stage.cost[k], stage.from[k] = local, -1
				continue
			}
			// Both idx and the previous row are ascending, so one shared cursor yields
			// the cheapest predecessor that still lies before candidate j.
			prev := &stages[g-1]
			for prev.scan < len(prev.idx) && prev.idx[prev.scan] < j {
				if prev.cost[prev.scan] < prev.best {
					prev.best, prev.bestK = prev.cost[prev.scan], prev.scan
				}
				prev.scan++
			}
			stage.cost[k], stage.from[k] = prev.best+local, prev.bestK
		}
		stages[g] = stage
	}

	chosen := make([]float64, numGaps)
	tail := stages[numGaps-1]
	bestK, bestCost := -1, math.Inf(1)
	for k, c := range tail.cost {
		if c < bestCost {
			bestCost, bestK = c, k
		}
	}
	if bestK < 0 || math.IsInf(bestCost, 1) {
		return proportionalTimings(boundaries, speechStart, speechEnd)
	}
	for g := numGaps - 1; g >= 0; g-- {
		chosen[g] = cands[stages[g].idx[bestK]].pos
		bestK = stages[g].from[bestK]
	}

	// Assemble: every word starts where the previous boundary landed, the first word
	// starts where the audio starts being speech and the last one ends where speech
	// stops, so leading and trailing silence belong to no word.
	timings := make([]WordTiming, len(boundaries))
	currentStart := speechStart
	for i, b := range boundaries {
		end := speechEnd
		if i < numGaps {
			end = chosen[i]
		}
		if end <= currentStart {
			end = math.Min(currentStart+windowDur, totalDuration)
		}
		if end > totalDuration {
			end = totalDuration
		}
		timings[i] = WordTiming{Word: b.Word, Start: currentStart, End: end}
		currentStart = end
	}
	return timings
}

// alignCandidate is one possible position for a word boundary: either a measured
// interior pause (owner -1) or the token-count prediction of the boundary that owns it.
type alignCandidate struct {
	pos   float64
	cost  float64 // pause-length discount in [0,1]; 1 means "not a pause at all"
	owner int
}

// alignStage is one boundary's row of the dynamic program. best/bestK/scan are the
// running prefix minimum of the cheapest monotone predecessor, consumed by the next row.
type alignStage struct {
	idx   []int
	cost  []float64
	from  []int
	scan  int
	best  float64
	bestK int
}

// maxAlignWindow bounds how many candidates a single boundary considers.
const maxAlignWindow = 64

// windowCandidates returns the ascending candidate indices within win seconds of pos
// that a boundary may use: the measured pauses plus its own prediction.
func windowCandidates(cands []alignCandidate, pos, win float64, ownIdx int) []int {
	lo := sort.Search(len(cands), func(i int) bool { return cands[i].pos >= pos-win })
	hi := sort.Search(len(cands), func(i int) bool { return cands[i].pos > pos+win })
	out := make([]int, 0, hi-lo+1)
	for j := lo; j < hi; j++ {
		if cands[j].owner >= 0 && j != ownIdx {
			continue // another boundary's escape hatch
		}
		out = append(out, j)
	}
	if ownIdx >= 0 && (ownIdx < lo || ownIdx >= hi) {
		out = append(out, ownIdx) // a boundary can always fall back to its prediction
	}
	if len(out) > maxAlignWindow { // even thinning keeps both ends of the window reachable
		stride := (len(out) + maxAlignWindow - 1) / maxAlignWindow
		thin := make([]int, 0, maxAlignWindow+1)
		for i := 0; i < len(out); i += stride {
			thin = append(thin, out[i])
		}
		if thin[len(thin)-1] != out[len(out)-1] {
			thin = append(thin, out[len(out)-1])
		}
		out = thin
	}
	return out
}

// proportionalTimings spreads the words over [start,end] by token count. It is the
// last resort when no pause could be aligned at all.
func proportionalTimings(boundaries []WordBoundary, start, end float64) []WordTiming {
	if len(boundaries) == 0 || end <= start {
		return nil
	}
	total := 0
	for _, b := range boundaries {
		if b.TokenCount > 0 {
			total += b.TokenCount
		} else {
			total++
		}
	}
	timings := make([]WordTiming, len(boundaries))
	cum, currentStart := 0, start
	for i, b := range boundaries {
		if b.TokenCount > 0 {
			cum += b.TokenCount
		} else {
			cum++
		}
		stop := start + (end-start)*float64(cum)/float64(total)
		if i == len(boundaries)-1 {
			stop = end
		}
		if stop <= currentStart {
			stop = math.Min(currentStart+(end-start)/float64(len(boundaries)), end)
		}
		timings[i] = WordTiming{Word: b.Word, Start: currentStart, End: stop}
		currentStart = stop
	}
	return timings
}

func p90(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	c := append([]float64(nil), v...)
	sort.Float64s(c)
	return c[(len(c)-1)*90/100]
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

// TokenizeWithVoice converts IPA phonemes into an ONNX ID sequence using the language
// code mapped to a specific voice. It is the plain-tokenization view of
// BuildTokenStream, so the ID sequence is exactly the one the alignment metadata of
// the same text is computed from.
func TokenizeWithVoice(text string, voice string) []int64 {
	return BuildTokenStream(text, voice).IDs
}
