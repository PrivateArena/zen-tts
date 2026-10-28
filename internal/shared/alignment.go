package shared

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Alignment support: per-word and per-phoneme timing derived from the model's own
// duration predictor instead of post-hoc audio analysis.
const (
	// DurationFrameSamples is the number of mono PCM samples covered by one frame of
	// the duration-predictor tensor of both Kokoro (StyleTTS2) and KittenTTS
	// (FastSpeech2-family) ONNX graphs: 600 samples = 25 ms at 24 kHz.
	DurationFrameSamples = 600

	// VADWindowMS is the RMS analysis window of SilenceBoundaryTimings: the resolution
	// the estimated (non duration-graph) timings are quantized to.
	VADWindowMS = 10

	// SpaceTokenID is the vocabulary ID of the space phoneme. It is the only
	// reliable word delimiter inside a single phonemization pass.
	SpaceTokenID = 16

	// TimingSourceDuration means the numbers come from the model's duration
	// predictor: exact, quantized to one model frame (see FrameMS).
	TimingSourceDuration = "duration_graph"
	// TimingSourceVAD means the numbers come from silence analysis of the PCM:
	// an estimate whose error is bounded by the pause structure, not the frame size.
	TimingSourceVAD = "vad_estimate"
	// TimingSourceUnknown is used for timings produced by an engine that does not
	// advertise how it derived them.
	TimingSourceUnknown = "estimated"
)

// PhonemeTiming holds one phoneme (or punctuation token, which carries the pause) of a
// synthesized utterance. Start/End are seconds from segment start; Frames is the number of
// model frames the predictor allotted the phoneme, already adjusted for the requested speed
// (Frames * DurationFrameSamples is its length in samples).
type PhonemeTiming struct {
	Phoneme   string  `json:"phoneme"`
	WordIndex int     `json:"word_index"`
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	Frames    int64   `json:"frames"`
}

// Alignment is the result of one synthesis pass: the audio plus the timing metadata
// that belongs to exactly that audio. Engines that can only estimate timings from
// the PCM leave Phonemes nil and set Source to TimingSourceVAD.
type Alignment struct {
	Samples    []float32       `json:"-"`
	SampleRate int             `json:"-"`
	Words      []WordTiming    `json:"words"`
	Phonemes   []PhonemeTiming `json:"phonemes,omitempty"`
	Source     string          `json:"source"`
	FrameMS    float64         `json:"frame_ms"`
}

// Duration returns the length of the aligned audio in seconds.
func (a Alignment) Duration() float64 {
	if a.SampleRate <= 0 {
		return 0
	}
	return float64(len(a.Samples)) / float64(a.SampleRate)
}

// WordSpan is the half-open token index range [Start,End) that a source word
// occupies inside a TokenStream.IDs slice (so it includes the BOS/EOS offset).
type WordSpan struct {
	Word  string
	Start int
	End   int
}

// TokenStream is the single source of truth for the token IDs handed to a model and
// for where the word boundaries are inside that sequence. Building both in one
// phonemization pass is what keeps the timing in lockstep with the audio: the token
// count produced per word here is by construction the count the model synthesized.
type TokenStream struct {
	// IDs are the ONNX token IDs, including the leading BOS and trailing EOS.
	IDs []int64
	// Labels holds the phoneme (or character, for the unmapped fallback) that
	// produced each ID. Empty for BOS/EOS. Same length as IDs.
	Labels []string
	// Words are the space-delimited groups found in IDs, in order.
	Words []WordSpan
	// WordsExact reports whether Words could be labelled with the real source
	// words; when false Word holds the phoneme text of the group instead.
	WordsExact bool
}

// BuildTokenStream phonemizes text once and returns the token IDs (bit-identical to
// TokenizeWithVoice), a label per token, and the per-word token spans obtained by
// splitting the stream on the space phoneme.
func BuildTokenStream(text, voice string) *TokenStream {
	lang := GetLanguageCode(voice)
	source := text
	if lang == "ja" {
		source = ConvertKanjiToHiragana(source)
	}

	ids := []int64{0} // BOS
	labels := []string{""}

	phonemes, err := Phonemize(source, lang)
	if err == nil && len(phonemes) > 0 {
		for _, p := range phonemes {
			if id, ok := Vocab[p]; ok {
				ids = append(ids, id)
				labels = append(labels, p)
				continue
			}
			// Combined IPA output: fall back to per-rune mapping, exactly as before.
			for _, r := range p {
				if id, ok := Vocab[string(r)]; ok {
					ids = append(ids, id)
					labels = append(labels, string(r))
				}
			}
		}
	} else {
		for _, r := range strings.ToLower(source) {
			if id, ok := Vocab[string(r)]; ok {
				ids = append(ids, id)
				labels = append(labels, string(r))
			}
		}
	}
	ids = append(ids, 0) // EOS
	labels = append(labels, "")

	ts := &TokenStream{IDs: ids, Labels: labels}
	ts.Words = groupBySpace(ids, labels)

	// Prefer the real source words as labels; the space split only lines up when it
	// produced exactly one group per whitespace-separated word.
	words := strings.Fields(source)
	if len(words) == len(ts.Words) {
		for i := range ts.Words {
			ts.Words[i].Word = words[i]
		}
		ts.WordsExact = len(words) > 0
	}
	return ts
}

// groupBySpace splits the body of a token stream (everything between BOS and EOS) into
// word spans on the space phoneme. Leading, trailing and repeated spaces yield no empty
// groups, and no span ever includes the BOS or the EOS frame.
func groupBySpace(ids []int64, labels []string) []WordSpan {
	if len(ids) < 2 {
		return nil
	}
	bodyEnd := len(ids) - 1 // index of EOS, which is not part of any word
	var spans []WordSpan
	start := -1
	for i := 1; i < bodyEnd; i++ {
		if ids[i] == SpaceTokenID {
			if start >= 0 {
				spans = append(spans, WordSpan{Word: joinLabels(labels, start, i), Start: start, End: i})
				start = -1
			}
			continue
		}
		if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		spans = append(spans, WordSpan{Word: joinLabels(labels, start, bodyEnd), Start: start, End: bodyEnd})
	}
	return spans
}

func joinLabels(labels []string, start, end int) string {
	var sb strings.Builder
	for _, l := range labels[start:end] {
		sb.WriteString(l)
	}
	return sb.String()
}

// WordBoundaries returns the per-word token counts that SilenceBoundaryTimings uses
// as its duration prior. ok is false when the space split could not be matched to
// the source words, i.e. the caller should fall back to per-word phonemization.
func (ts *TokenStream) WordBoundaries() (boundaries []WordBoundary, ok bool) {
	if ts == nil || len(ts.Words) == 0 {
		return nil, false
	}
	boundaries = make([]WordBoundary, 0, len(ts.Words))
	for _, w := range ts.Words {
		boundaries = append(boundaries, WordBoundary{Word: w.Word, TokenCount: w.End - w.Start})
	}
	return boundaries, ts.WordsExact
}

// CharCountBoundaries builds a token-count prior from character counts. It is for
// engines whose phonemizer and token vocabulary are not visible here (Piper), where
// character count is the cheapest monotone proxy for spoken duration.
func CharCountBoundaries(text string) []WordBoundary {
	words := strings.Fields(text)
	boundaries := make([]WordBoundary, 0, len(words))
	for _, w := range words {
		n := utf8.RuneCountInString(w)
		if n < 1 {
			n = 1
		}
		boundaries = append(boundaries, WordBoundary{Word: w, TokenCount: n})
	}
	return boundaries
}

// TimingsFromDurations converts a per-token duration tensor (in model frames) into
// word and phoneme timings. durations must come from the same token stream as ts and
// from the same synthesis pass as the audio:
//   - modelSamples is the sample count the graph produced for this token stream before
//     any trailing-silence trim, which is what the duration tensor sums to;
//   - audioSamples is the sample count actually being returned, so trimming can never
//     leave the last word claiming time that does not exist.
func TimingsFromDurations(durations []int64, ts *TokenStream, sampleRate, modelSamples, audioSamples int) (words []WordTiming, phonemes []PhonemeTiming, frameMS float64, err error) {
	if ts == nil || len(ts.IDs) == 0 || sampleRate <= 0 || audioSamples <= 0 {
		return nil, nil, 0, fmt.Errorf("no token stream or audio to align")
	}

	// Tolerant of graphs that omit the BOS/EOS frames (mapped below).
	if len(durations) != len(ts.IDs) && len(durations) != len(ts.IDs)-2 {
		return nil, nil, 0, fmt.Errorf("duration tensor has %d entries, expected %d or %d tokens",
			len(durations), len(ts.IDs), len(ts.IDs)-2)
	}

	// Frame count per token, then the prefix sum: cum[i] is the start of token i.
	tokenFrames := make([]int64, len(ts.IDs))
	switch len(durations) {
	case len(ts.IDs):
		for i, d := range durations {
			tokenFrames[i] = positiveFrames(d)
		}
	case len(ts.IDs) - 2:
		// Body-only tensor: the graph did not emit BOS/EOS frames, so the leading and
		// trailing pauses are absorbed into the first and last word.
		for i := 1; i < len(ts.IDs)-1; i++ {
			tokenFrames[i] = positiveFrames(durations[i-1])
		}
	}
	var totalFrames int64
	for _, f := range tokenFrames {
		totalFrames += f
	}
	if totalFrames <= 0 {
		return nil, nil, 0, fmt.Errorf("duration tensor sums to %d frames", totalFrames)
	}

	// The graphs are exact: sum(frames)*DurationFrameSamples == sample count. Trust
	// the constant when the audio agrees with it, otherwise scale to the audio we
	// were actually given (a different model version, or a different hop size).
	samplesPerFrame := float64(DurationFrameSamples)
	expected := float64(totalFrames) * DurationFrameSamples
	reference := modelSamples
	if reference <= 0 {
		reference = audioSamples
	}
	if float64(reference) > expected || float64(reference) < expected*0.98 {
		samplesPerFrame = float64(reference) / float64(totalFrames)
	}
	frameMS = samplesPerFrame / float64(sampleRate) * 1000.0
	audioSeconds := float64(audioSamples) / float64(sampleRate)
	sec := func(frames int64) float64 {
		if t := float64(frames) * samplesPerFrame / float64(sampleRate); t < audioSeconds {
			return t
		}
		return audioSeconds
	}

	cum := make([]int64, len(ts.IDs)+1)
	for i, f := range tokenFrames {
		cum[i+1] = cum[i] + f
	}

	for _, w := range ts.Words {
		if w.End <= w.Start {
			continue
		}
		words = append(words, WordTiming{
			Word:  w.Word,
			Start: sec(cum[w.Start]),
			End:   sec(cum[w.End]),
		})
	}
	// The spans are ordered, so one cursor walks them while the tokens stream past.
	span := 0
	for i := 1; i < len(ts.IDs); i++ {
		for span < len(ts.Words) && i >= ts.Words[span].End {
			span++
		}
		if ts.IDs[i] == SpaceTokenID || ts.Labels[i] == "" {
			continue // spaces carry no viseme, BOS/EOS have no label
		}
		wordIndex := -1
		if span < len(ts.Words) && i >= ts.Words[span].Start {
			wordIndex = span
		}
		phonemes = append(phonemes, PhonemeTiming{
			Phoneme:   ts.Labels[i],
			WordIndex: wordIndex,
			Start:     sec(cum[i]),
			End:       sec(cum[i+1]),
			Frames:    tokenFrames[i],
		})
	}
	return words, phonemes, frameMS, nil
}

func positiveFrames(d int64) int64 {
	if d > 0 {
		return d
	}
	return 0
}
