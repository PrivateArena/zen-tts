package shared

import (
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// Synthetic PCM helpers
// ---------------------------------------------------------------------------

const testRate = 24000

// speech renders a constant-loudness band-limited tone: a stable RMS envelope, which is
// what the windowed detector measures.
func speech(seconds float64, amp float64) []float32 {
	n := int(seconds * float64(testRate))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(amp * math.Sin(2*math.Pi*120.0*float64(i)/float64(testRate)) *
			(0.75 + 0.25*math.Sin(2*math.Pi*7.0*float64(i)/float64(testRate))))
	}
	return out
}

func quiet(seconds float64) []float32 {
	return make([]float32, int(seconds*float64(testRate)))
}

func concat(parts ...[]float32) []float32 {
	var out []float32
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func words(n int) []WordBoundary {
	out := make([]WordBoundary, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, WordBoundary{Word: string(rune('a' + i%26)), TokenCount: 1})
	}
	return out
}

// ---------------------------------------------------------------------------
// Characterization: with no silence anywhere the detector cannot invent structure, so
// the timings must be exactly the token-count proportions over the whole utterance.
// This pins the numbers the previous implementation produced for that case.
// ---------------------------------------------------------------------------

func TestSilenceBoundaryTimingsNoSilenceIsProportional(t *testing.T) {
	samples := speech(1.0, 0.7)
	boundaries := []WordBoundary{{"a", 1}, {"b", 2}, {"c", 3}, {"d", 4}}

	got := SilenceBoundaryTimings(samples, testRate, boundaries)
	if len(got) != 4 {
		t.Fatalf("got %d timings, want 4", len(got))
	}
	want := []WordTiming{
		{Word: "a", Start: 0.0, End: 0.1},
		{Word: "b", Start: 0.1, End: 0.3},
		{Word: "c", Start: 0.3, End: 0.6},
		{Word: "d", Start: 0.6, End: 1.0},
	}
	for i := range want {
		if math.Abs(got[i].Start-want[i].Start) > 1e-6 || math.Abs(got[i].End-want[i].End) > 1e-6 {
			t.Errorf("word %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// ---------------------------------------------------------------------------
// The two headline defects the silence heuristics used to have.
// ---------------------------------------------------------------------------

// TestSilenceBoundaryTimingsClampsLeadingAndTrailingSilence guards the "boundary at 0.0
// inside 0.4 s of leading silence" failure: leading and trailing silence belong to no
// word, and the first real boundary must not be swallowed by it.
func TestSilenceBoundaryTimingsClampsLeadingAndTrailingSilence(t *testing.T) {
	samples := concat(
		quiet(0.4), speech(0.2, 0.7), quiet(0.15), speech(0.25, 0.7), quiet(0.1),
	) // 1.2 s
	boundaries := []WordBoundary{{"the", 3}, {"word", 3}}

	got := SilenceBoundaryTimings(samples, testRate, boundaries)
	if len(got) != 2 {
		t.Fatalf("got %d timings, want 2", len(got))
	}
	if got[0].Start < 0.35 || got[0].Start > 0.45 {
		t.Errorf("first word starts at %.3f, want the 0.400 s end of the leading silence", got[0].Start)
	}
	// The only interior pause spans 0.600-0.750.
	if got[0].End < 0.60 || got[0].End > 0.76 {
		t.Errorf("boundary landed at %.3f, outside the only pause (0.600-0.750)", got[0].End)
	}
	// The trailing silence is 1.000-1.100, so the last word must stop at 1.0 and not
	// claim the 100 ms of nothing that closes the utterance.
	if got[1].End > 1.02 || got[1].End < 0.97 {
		t.Errorf("last word ends at %.3f, want the 1.000 s start of the trailing silence", got[1].End)
	}
}

// TestSilenceBoundaryTimingsAlignsAllPausesMonotonically is the regression guard for the
// per-gap independent search: every pause must be attributed to its own word pair, in
// word order, and a boundary that would fall behind its predecessor must never happen.
func TestSilenceBoundaryTimingsAlignsAllPausesMonotonically(t *testing.T) {
	samples := concat(
		speech(0.10, 0.7), // word a
		quiet(0.10),       // long pause: a|b
		speech(0.25, 0.7), // word b
		quiet(0.02),       // short pause: b|c
		speech(0.23, 0.7), // word c
		quiet(0.05),       // medium pause: c|d
		speech(0.25, 0.7), // word d
	)
	boundaries := []WordBoundary{{"a", 1}, {"b", 1}, {"c", 1}, {"d", 1}}

	got := SilenceBoundaryTimings(samples, testRate, boundaries)
	if len(got) != 4 {
		t.Fatalf("got %d timings, want 4", len(got))
	}
	wantCentres := []float64{0.15, 0.46, 0.725}
	for i, want := range wantCentres {
		if math.Abs(got[i].End-want) > 0.05 {
			t.Errorf("boundary %d (pair %s|%s) at %.3f, want the measured pause at %.3f (all: %v)",
				i, boundaries[i].Word, boundaries[i+1].Word, got[i].End, want, ends(got))
		}
	}
	assertMonotone(t, got, float64(len(samples))/testRate)
}

// ---------------------------------------------------------------------------
// Invariants that must hold for any input.
// ---------------------------------------------------------------------------

func TestSilenceBoundaryTimingsInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(20261028))
	for trial := 0; trial < 60; trial++ {
		nWords := 1 + rng.Intn(12)
		parts := make([][]float32, 0, 2*nWords+2)
		if rng.Intn(2) == 0 {
			parts = append(parts, quiet(rng.Float64()*0.4))
		}
		for w := 0; w < nWords; w++ {
			parts = append(parts, speech(0.05+rng.Float64()*0.3, 0.3+0.6*rng.Float64()))
			parts = append(parts, quiet(rng.Float64()*0.2))
		}
		parts = append(parts, quiet(rng.Float64()*0.4))
		samples := concat(parts...)
		if len(samples) < testRate/4 {
			continue
		}

		boundaries := make([]WordBoundary, 0, nWords)
		for w := 0; w < nWords; w++ {
			boundaries = append(boundaries, WordBoundary{Word: string(rune('a' + w)), TokenCount: 1 + rng.Intn(8)})
		}

		got := SilenceBoundaryTimings(samples, testRate, boundaries)
		duration := float64(len(samples)) / testRate
		if len(got) != len(boundaries) {
			t.Fatalf("trial %d: got %d timings for %d words", trial, len(got), len(boundaries))
		}
		for i, w := range got {
			if w.Word != boundaries[i].Word {
				t.Errorf("trial %d: timing %d lost its word label (%q vs %q)", trial, i, w.Word, boundaries[i].Word)
			}
			if w.Start < 0 || w.End > duration+1e-9 {
				t.Errorf("trial %d: timing %+v escapes [0, %v]", trial, w, duration)
			}
			if w.End <= w.Start {
				t.Errorf("trial %d: zero/negative-width word %+v", trial, w)
			}
			if i > 0 && math.Abs(w.Start-got[i-1].End) > 1e-9 {
				t.Errorf("trial %d: word %d starts at %v but the previous ends at %v", trial, i, w.Start, got[i-1].End)
			}
			// Timings must be strictly ordered: the old fallback invented equal slices
			// whenever the independent per-gap search produced a backwards boundary.
			if got[i].End <= got[i].Start {
				t.Errorf("trial %d: non-monotonic boundary at %+v", trial, w)
			}
		}
	}
}

func ends(got []WordTiming) []float64 {
	out := make([]float64, len(got))
	for i, w := range got {
		out[i] = w.End
	}
	return out
}

func assertMonotone(t *testing.T, got []WordTiming, duration float64) {
	t.Helper()
	for i, w := range got {
		if w.End <= w.Start {
			t.Errorf("word %d (%q) has non-positive width: %+v", i, w.Word, w)
		}
		if w.End > duration+1e-9 {
			t.Errorf("word %d (%q) ends at %v, past the audio end %v", i, w.Word, w.End, duration)
		}
		if i > 0 && w.Start < got[i-1].End {
			t.Errorf("word %d starts (%v) before the previous word ended (%v)", i, w.Start, got[i-1].End)
		}
	}
}

// TestSilenceBoundaryTimingsQuietWordKeepsWidth: a whispered word between loud ones must
// still get a span, i.e. the detector may not collapse it into the surrounding pauses.
func TestSilenceBoundaryTimingsQuietWordKeepsWidth(t *testing.T) {
	samples := concat(speech(0.3, 1.0), quiet(0.05), speech(0.3, 0.003), quiet(0.05), speech(0.3, 1.0))
	boundaries := []WordBoundary{{"loud", 2}, {"hushed", 2}, {"loud2", 2}}

	got := SilenceBoundaryTimings(samples, testRate, boundaries)
	if len(got) != 3 {
		t.Fatalf("got %d timings, want 3", len(got))
	}
	duration := float64(len(samples)) / testRate
	assertMonotone(t, got, duration)
	minWidth := 0.15 * duration / 3
	if got[1].End-got[1].Start < minWidth {
		t.Errorf("quiet word got %.3f s of the %.3f s utterance (want at least %.3f): %+v",
			got[1].End-got[1].Start, duration, minWidth, got)
	}
}

func TestSilenceBoundaryTimingsDegenerateInputs(t *testing.T) {
	boundaries := []WordBoundary{{"a", 1}, {"b", 1}}

	if got := SilenceBoundaryTimings(nil, testRate, boundaries); got != nil {
		t.Errorf("no samples must give nil, got %+v", got)
	}
	if got := SilenceBoundaryTimings(speech(0.5, 0.5), testRate, nil); got != nil {
		t.Errorf("no words must give nil, got %+v", got)
	}
	if got := SilenceBoundaryTimings(speech(0.5, 0.5), 0, boundaries); got != nil {
		t.Errorf("invalid sample rate must give nil, got %+v", got)
	}
	// Shorter than one analysis window: still a well-formed answer.
	got := SilenceBoundaryTimings(speech(0.001, 0.5), testRate, boundaries)
	if len(got) != 2 {
		t.Fatalf("got %d timings for a sub-window utterance", len(got))
	}
	assertMonotone(t, got, 0.001)

	one := SilenceBoundaryTimings(concat(quiet(0.2), speech(0.6, 0.8), quiet(0.2)), testRate, []WordBoundary{{"solo", 3}})
	if len(one) != 1 {
		t.Fatalf("got %d timings, want 1", len(one))
	}
	if one[0].Start < 0.19 || one[0].End > 0.81 {
		t.Errorf("single word should span the speech only, got %+v", one[0])
	}
}

func TestSilenceBoundaryTimingsAllSilenceStillSpreadsWords(t *testing.T) {
	samples := quiet(1.0)
	boundaries := []WordBoundary{{"a", 1}, {"b", 3}, {"c", 1}}

	got := SilenceBoundaryTimings(samples, testRate, boundaries)
	if len(got) != 3 {
		t.Fatalf("got %d timings, want 3", len(got))
	}
	assertMonotone(t, got, 1.0)
	if got[2].End != 1.0 {
		t.Errorf("utterance with no detectable speech must still cover its full duration, got %+v", got)
	}
}
