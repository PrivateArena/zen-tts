package inflect

import (
	"regexp"
	"strings"
	"unicode"
)

var symbols = []string{
	"_", ";", ":", ",", ".", "!", "?", "—", "…", "\"", "(", ")", "“", "”", " ",
	"̃", "ʣ", "ʥ", "ʦ", "ʨ", "ᵝ", "ꭧ",
	"A", "I", "O", "Q", "S", "T", "W", "Y", "ᵊ",
	"a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
	"ɑ", "ɐ", "ɒ", "æ", "β", "ɔ", "ɕ", "ç", "ɖ", "ð", "ʤ", "ə", "ɚ", "ɛ", "ɜ", "ɟ", "ɡ", "ɥ", "ɨ", "ɪ", "ʝ", "ɯ", "ɰ", "ŋ", "ɳ", "ɲ", "ɴ", "ø", "ɸ", "θ", "œ", "ɹ", "ɾ", "ɻ", "ʁ", "ɽ", "ʂ", "ʃ", "ʈ", "ʧ", "ʊ", "ʋ", "ʌ", "ɣ", "ɤ", "χ", "ʎ", "ʒ", "ʔ", "ˈ", "ˌ", "ː", "ʰ", "ʲ", "↓", "→", "↗", "↘", "ᵻ",
}

var symbolToID map[string]int64

func init() {
	symbolToID = make(map[string]int64, len(symbols))
	for i, s := range symbols {
		symbolToID[s] = int64(i)
	}
}

func PhonemesToTokens(phonemes []string, vocab map[string]int64) []int64 {
	var seq []int64
	for _, p := range phonemes {
		if id, ok := vocab[p]; ok {
			seq = append(seq, id)
		} else {
			for _, r := range p {
				if id, ok := vocab[string(r)]; ok {
					seq = append(seq, id)
				}
			}
		}
	}
	withBlanks := make([]int64, len(seq)*2+1)
	for i, id := range seq {
		withBlanks[1+i*2] = id
	}
	return withBlanks
}

var (
	wordOverrides = map[string]string{
		"vs":   "versus",
		"etc":  "et cetera",
		"diy":  "do it yourself",
		"asap": "as soon as possible",
		"aka":  "also known as",
		"ie":   "that is",
		"eg":   "for example",
		"dept": "department",
		"apt":  "apartment",
		"est":  "estimated",
		"temp": "temporary",
		"info": "information",
		"ad":   "advertisement",
	}

	abbreviations = map[string]string{
		"dr.":   "doctor",
		"mr.":   "mister",
		"mrs.":  "misses",
		"ms.":   "miss",
		"prof.": "professor",
		"sr.":   "senior",
		"jr.":   "junior",
		"st.":   "saint",
		"ave.":  "avenue",
		"blvd.": "boulevard",
		"dept.": "department",
		"govt.": "government",
		"inc.":  "incorporated",
		"ltd.":  "limited",
		"co.":   "company",
		"corp.": "corporation",
		"etc.":  "et cetera",
		"vs.":   "versus",
		"e.g.":  "for example",
		"i.e.":  "that is",
	}

	punctTranslation = strings.NewReplacer(
		"…", ".",
		"—", ",",
		"\u2013", "-",
		"\u2014", "-",
		"\u2018", "'",
		"\u2019", "'",
		"\u201c", "\"",
		"\u201d", "\"",
		"\u00ab", "\"",
		"\u00bb", "\"",
		"\u00a0", " ",
	)
)

var (
	reMultipleSpaces = regexp.MustCompile(`\s+`)
	reURL            = regexp.MustCompile(`https?://\S+`)
	reEmail          = regexp.MustCompile(`\S+@\S+\.\S+`)
)

func NormalizeText(text string) string {
	text = reURL.ReplaceAllString(text, " URL ")
	text = reEmail.ReplaceAllString(text, " EMAIL ")
	text = punctTranslation.Replace(text)
	text = reMultipleSpaces.ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)

	for src, dst := range wordOverrides {
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(src) + `\b`)
		text = re.ReplaceAllString(text, dst)
	}
	for src, dst := range abbreviations {
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(src) + `\b`)
		text = re.ReplaceAllString(text, dst)
	}
	text = reMultipleSpaces.ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

func SplitText(text string, limit int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if len([]rune(text)) <= limit {
		return []string{text}
	}

	sentinels := []string{". ", "! ", "? ", "; ", ": ", ", "}
	segments := []string{text}
	for _, sentinel := range sentinels {
		var next []string
		for _, seg := range segments {
			if len([]rune(seg)) <= limit {
				next = append(next, seg)
				continue
			}
			parts := splitAtLast(seg, sentinel, limit)
			next = append(next, parts...)
		}
		segments = next
	}

	var result []string
	for _, seg := range segments {
		if len([]rune(seg)) <= limit {
			result = append(result, seg)
		} else {
			words := strings.Fields(seg)
			var buf []string
			runes := 0
			for _, w := range words {
				wlen := len([]rune(w))
				if runes+wlen+len(buf) > limit && len(buf) > 0 {
					result = append(result, strings.Join(buf, " "))
					buf = nil
					runes = 0
				}
				buf = append(buf, w)
				runes += wlen
			}
			if len(buf) > 0 {
				result = append(result, strings.Join(buf, " "))
			}
		}
	}
	return result
}

func splitAtLast(s string, sentinel string, limit int) []string {
	runes := []rune(s)
	if len(runes) <= limit {
		return []string{s}
	}
	search := []rune(sentinel)
	cut := -1
	for i := limit - 1; i >= limit/2; i-- {
		if i+len(search) <= len(runes) {
			match := true
			for j := 0; j < len(search); j++ {
				if runes[i+j] != search[j] {
					match = false
					break
				}
			}
			if match {
				cut = i + len(search)
				break
			}
		}
	}
	if cut < 0 {
		cut = limit
	}
	return []string{string(runes[:cut]), string(runes[cut:])}
}

func BoundaryPauseSeconds(prevChunk string) float64 {
	last := strings.TrimRightFunc(prevChunk, unicode.IsSpace)
	if len(last) == 0 {
		return 0.08
	}
	switch last[len(last)-1] {
	case '?':
		return 0.28
	case '!':
		return 0.24
	case '.':
		return 0.22
	case ';':
		return 0.16
	case ':':
		return 0.13
	case ',':
		return 0.09
	default:
		return 0.08
	}
}

func EdgeFade(waveform []float32, sampleRate int, ms float64) []float32 {
	frames := int(float64(sampleRate) * ms / 1000.0)
	if frames <= 0 || frames > len(waveform)/2 {
		return waveform
	}
	out := make([]float32, len(waveform))
	copy(out, waveform)
	for i := 0; i < frames; i++ {
		ramp := float32(i) / float32(frames)
		out[i] *= ramp
		out[len(out)-1-i] *= ramp
	}
	return out
}
