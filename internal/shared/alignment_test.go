package shared

import (
	"math"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// BuildTokenStream: one phonemization pass, tokens + labels + word spans.
// These run through the character-mapping fallback (no ./piper next to the test
// binary), which exercises the same grouping code as the IPA path.
// ---------------------------------------------------------------------------

func TestBuildTokenStreamFallsBackToCharMapping(t *testing.T) {
	ts := BuildTokenStream("hello world", "en-us")

	if len(ts.IDs) < 3 {
		t.Fatalf("expected BOS + body + EOS, got %d ids", len(ts.IDs))
	}
	if ts.IDs[0] != 0 || ts.IDs[len(ts.IDs)-1] != 0 {
		t.Errorf("BOS/EOS must bracket the stream: %v", ts.IDs)
	}
	if len(ts.Labels) != len(ts.IDs) {
		t.Fatalf("labels/tokens length mismatch: %d vs %d", len(ts.Labels), len(ts.IDs))
	}
	if ts.Labels[0] != "" || ts.Labels[len(ts.Labels)-1] != "" {
		t.Errorf("BOS/EOS must have empty labels, got %q / %q", ts.Labels[0], ts.Labels[len(ts.Labels)-1])
	}
	if !ts.WordsExact {
		t.Errorf("two whitespace words should label the two space groups exactly")
	}
	want := []string{"hello", "world"}
	if len(ts.Words) != len(want) {
		t.Fatalf("got %d word spans, want %d (%v)", len(ts.Words), len(want), ts.Words)
	}
	for i, w := range ts.Words {
		if w.Word != want[i] {
			t.Errorf("span %d label = %q, want %q", i, w.Word, want[i])
		}
		if w.Start >= w.End {
			t.Errorf("span %d is empty: %+v", i, w)
		}
		for j := w.Start; j < w.End; j++ {
			if ts.IDs[j] == SpaceTokenID {
				t.Errorf("span %d contains a space token at %d", i, j)
			}
		}
	}
	if ts.Words[0].Start != 1 {
		t.Errorf("first word must start right after BOS, got %d", ts.Words[0].Start)
	}
	if ts.Words[len(ts.Words)-1].End != len(ts.IDs)-1 {
		t.Errorf("last word must end right before EOS, got %d of %d ids", ts.Words[len(ts.Words)-1].End, len(ts.IDs))
	}
}

func TestBuildTokenStreamSpanInvariants(t *testing.T) {
	for _, text := range []string{
		"one",
		"one two three",
		"  padded   gaps  ",
		"",
		"...",
		"tabs\tand\nnewlines here",
	} {
		ts := BuildTokenStream(text, "en-us")
		prevEnd := 1
		for i, w := range ts.Words {
			if w.Start < 1 || w.End > len(ts.IDs)-1 {
				t.Fatalf("text %q span %d escapes the body: %+v (ids=%d)", text, i, w, len(ts.IDs))
			}
			if w.Start < prevEnd {
				t.Fatalf("text %q spans overlap/are unordered: %+v after end %d", text, w, prevEnd)
			}
			prevEnd = w.End
			if ts.WordsExact && strings.TrimSpace(w.Word) == "" {
				t.Errorf("text %q: exact-labelled span %d is blank (%q)", text, i, w.Word)
			}
		}
		if words := strings.Fields(text); len(words) != len(ts.Words) && ts.WordsExact {
			t.Errorf("text %q: WordsExact set but %d fields vs %d spans", text, len(words), len(ts.Words))
		}
	}
}

func TestBuildTokenStreamAbsorbsRepeatedSpaces(t *testing.T) {
	// A doubled space is one delimiter too many in the text, not a missing word: the
	// empty group is dropped so the three source words still label the three spans.
	ts := BuildTokenStream("al al  al", "en-us")
	if !ts.WordsExact {
		t.Fatalf("expected exact labelling across a doubled space, got %v", ts.Words)
	}
	for _, w := range ts.Words {
		if w.Word != "al" {
			t.Errorf("bad label %q in %v", w.Word, ts.Words)
		}
	}
}

func TestBuildTokenStreamFallsBackToPhonemeLabels(t *testing.T) {
	// A tab is not in the vocabulary, so "a\tb" is one phoneme group for two source
	// words: the labels then fall back to the phoneme text instead of lying.
	ts := BuildTokenStream("al\tal", "en-us")
	if ts.WordsExact {
		t.Fatalf("expected inexact labelling when group and word counts differ, got %v", ts.Words)
	}
	if _, ok := Vocab["\t"]; ok {
		t.Fatalf("test premise broken: tab is in the vocabulary")
	}
	for _, w := range ts.Words {
		if w.Word == "" {
			t.Errorf("empty group label in %v", ts.Words)
		}
	}
}

func TestTokenStreamWordBoundaries(t *testing.T) {
	ts := BuildTokenStream("hello world", "en-us")
	boundaries, ok := ts.WordBoundaries()
	if !ok {
		t.Errorf("ok = false for a stream whose groups match the words")
	}
	if len(boundaries) != 2 {
		t.Fatalf("got %d boundaries, want 2", len(boundaries))
	}
	total := 0
	for i, b := range boundaries {
		if b.TokenCount != ts.Words[i].End-ts.Words[i].Start {
			t.Errorf("boundary %d count = %d, want %d", i, b.TokenCount, ts.Words[i].End-ts.Words[i].Start)
		}
		if b.TokenCount == 0 {
			t.Errorf("boundary %d has zero tokens: %+v", i, b)
		}
		total += b.TokenCount
	}
	if want := len(ts.IDs) - 3; total > want { // body minus the inter-word spaces
		t.Errorf("boundary counts sum to %d, more than the %d non-space body tokens", total, want)
	}

	empty := &TokenStream{}
	if got, ok := empty.WordBoundaries(); got != nil || ok {
		t.Errorf("empty stream must yield no boundaries, got %+v/%v", got, ok)
	}
}

func TestCharCountBoundaries(t *testing.T) {
	got := CharCountBoundaries("hi there 你好")
	want := []WordBoundary{{"hi", 2}, {"there", 5}, {"你好", 2}}
	if len(got) != len(want) {
		t.Fatalf("got %d boundaries, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Word != want[i].Word || got[i].TokenCount != want[i].TokenCount {
			t.Errorf("boundary %d = %+v, want %+v", i, got[i], want[i])
		}
	}
	if got := CharCountBoundaries("   "); len(got) != 0 {
		t.Errorf("whitespace-only text must yield no boundaries, got %+v", got)
	}
}

// ---------------------------------------------------------------------------
// TimingsFromDurations: the frame-exact mapping from duration predictor to seconds.
// ---------------------------------------------------------------------------

// fakeStream is "hello world" as the model sees it: BOS, 5 phonemes, space, 5
// phonemes, EOS.
func fakeStream() *TokenStream {
	labels := []string{"", "h", "ɛ", "l", "oʊ", " ", "w", "ɜː", "l", "d", ""}
	ids := make([]int64, len(labels))
	for i, l := range labels {
		switch {
		case l == "":
			ids[i] = 0
		case l == " ":
			ids[i] = SpaceTokenID
		default:
			ids[i] = int64(40 + i)
		}
	}
	return &TokenStream{
		IDs:        ids,
		Labels:     labels,
		Words:      []WordSpan{{Word: "hello", Start: 1, End: 5}, {Word: "world", Start: 6, End: 10}},
		WordsExact: true,
	}
}

func TestTimingsFromDurationsExactFrames(t *testing.T) {
	ts := fakeStream()
	// 1 frame BOS, then 2,1,3,1 for "hello", 4 for the space, 1,1,1,2 for "world", 1 EOS.
	durations := []int64{1, 2, 1, 3, 1, 4, 1, 1, 1, 2, 1}
	total := int64(0)
	for _, d := range durations {
		total += d
	}

	words, phonemes, frameMS, err := TimingsFromDurations(durations, ts, 24000, int(total)*DurationFrameSamples, int(total)*DurationFrameSamples)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if frameMS != 25.0 {
		t.Errorf("frameMS = %v, want 25 (600 samples at 24 kHz)", frameMS)
	}
	if len(words) != 2 {
		t.Fatalf("got %d words, want 2", len(words))
	}

	// "hello" covers tokens 1..5 -> frames 1..9 of the prefix sum.
	assertSec(t, "hello.Start", words[0].Start, 1*0.025)
	assertSec(t, "hello.End", words[0].End, 8*0.025)
	assertSec(t, "world.Start", words[1].Start, 12*0.025)
	assertSec(t, "world.End", words[1].End, 17*0.025)
	if words[0].Word != "hello" || words[1].Word != "world" {
		t.Errorf("word labels lost: %v", words)
	}

	// Spaces carry no viseme, so exactly the 8 phonemes of both words remain.
	if len(phonemes) != 8 {
		t.Fatalf("got %d phonemes, want 8 (spaces must be skipped)", len(phonemes))
	}
	for _, p := range phonemes {
		if p.Phoneme == "" || p.Phoneme == " " {
			t.Errorf("empty phoneme in %+v", p)
		}
		if p.WordIndex < 0 || p.WordIndex >= len(words) {
			t.Errorf("phoneme %q has no word index: %+v", p.Phoneme, p)
		}
		if p.End <= p.Start {
			t.Errorf("phoneme %q is zero-width: %+v", p.Phoneme, p)
		}
		if p.Frames <= 0 {
			t.Errorf("phoneme %q has no frames: %+v", p.Phoneme, p)
		}
	}
	if phonemes[0].WordIndex != 0 || phonemes[7].WordIndex != 1 {
		t.Errorf("phoneme word attribution wrong: %+v", phonemes)
	}
	assertSec(t, "first phoneme start", phonemes[0].Start, 1*0.025)
}

func TestTimingsFromDurationsBodyOnlyTensor(t *testing.T) {
	ts := fakeStream()
	durations := []int64{2, 1, 3, 1, 4, 1, 1, 1, 2} // no BOS/EOS frames
	words, _, _, err := TimingsFromDurations(durations, ts, 24000, 16*DurationFrameSamples, 16*DurationFrameSamples)
	if err != nil {
		t.Fatalf("body-only duration tensor must be accepted: %v", err)
	}
	assertSec(t, "hello.Start", words[0].Start, 0)
	assertSec(t, "hello.End", words[0].End, 7*0.025)
	assertSec(t, "world.End", words[1].End, 16*0.025)
}

func TestTimingsFromDurationsClampsToTrimmedAudio(t *testing.T) {
	ts := fakeStream()
	durations := []int64{1, 2, 1, 3, 1, 4, 1, 1, 1, 2, 1}
	// 18 frames of audio, of which the engine trimmed the last 4 as exact zeros.
	trimmed := 14 * DurationFrameSamples
	words, phonemes, _, err := TimingsFromDurations(durations, ts, 24000, 18*DurationFrameSamples, trimmed)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	max := float64(trimmed) / 24000.0
	if words[len(words)-1].End > max+1e-9 {
		t.Errorf("last word ends at %v, past the trimmed audio duration %v", words[len(words)-1].End, max)
	}
	for _, p := range phonemes {
		if p.End > max+1e-9 {
			t.Errorf("phoneme %q ends at %v, past %v", p.Phoneme, p.End, max)
		}
	}
}

func TestTimingsFromDurationsScalesWhenFramesDisagree(t *testing.T) {
	ts := fakeStream()
	durations := []int64{1, 2, 1, 3, 1, 4, 1, 1, 1, 2, 1} // 19 frames
	// A graph whose hop is not 600 samples: timings must stretch to the real audio.
	const hop = 512
	const frames = 18 // == sum(durations)
	words, _, frameMS, err := TimingsFromDurations(durations, ts, 24000, frames*hop, frames*hop)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := float64(hop) / 24.0; math.Abs(frameMS-want) > 1e-6 {
		t.Errorf("frameMS = %v, want %v", frameMS, want)
	}
	assertSec(t, "last word end", words[1].End, float64(17*hop)/24000.0)
}

func TestTimingsFromDurationsRejectsBadInput(t *testing.T) {
	ts := fakeStream()
	good := []int64{1, 2, 1, 3, 1, 4, 1, 1, 1, 2, 1}

	if _, _, _, err := TimingsFromDurations([]int64{1, 2, 3}, ts, 24000, 1000, 1000); err == nil {
		t.Errorf("wrong-length duration tensor must be rejected")
	}
	if _, _, _, err := TimingsFromDurations(good, nil, 24000, 1000, 1000); err == nil {
		t.Errorf("nil token stream must be rejected")
	}
	if _, _, _, err := TimingsFromDurations(good, ts, 0, 1000, 1000); err == nil {
		t.Errorf("zero sample rate must be rejected")
	}
	if _, _, _, err := TimingsFromDurations(good, ts, 24000, 1000, 0); err == nil {
		t.Errorf("empty audio must be rejected")
	}
	if _, _, _, err := TimingsFromDurations([]int64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, ts, 24000, 1000, 1000); err == nil {
		t.Errorf("all-zero durations must be rejected rather than divide by zero")
	}
	// Negative frames must not produce negative times.
	neg := []int64{-1, -2, -3, -1, -1, -1, -1, -1, -1, -1, -1}
	if _, _, _, err := TimingsFromDurations(neg, ts, 24000, 1000, 1000); err == nil {
		t.Errorf("all-negative durations must be rejected")
	}
}

func assertSec(t *testing.T, what string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %.6f, want %.6f", what, got, want)
	}
}
