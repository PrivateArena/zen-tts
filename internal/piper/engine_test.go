package piper

import (
	"testing"

	"zen-tts/internal/shared"
)

// TestPiperTimingsWithoutASession: the timing entry points must fail like Synthesize does,
// never return a half-built alignment or panic on a closed synthesizer.
func TestPiperTimingsWithoutASession(t *testing.T) {
	e := &PiperEngine{}

	if _, _, _, err := e.SynthesizeWithTimings("hello world", "0", 1.0); err == nil {
		t.Error("SynthesizeWithTimings on a closed engine must report an error")
	}
	a, err := e.SynthesizeAligned("hello world", "0", 1.0)
	if err == nil {
		t.Error("SynthesizeAligned on a closed engine must report an error")
	}
	if len(a.Samples) != 0 || len(a.Words) != 0 {
		t.Errorf("a failed alignment must be empty, got %d samples and %d words", len(a.Samples), len(a.Words))
	}
}

// TestPiperEstimatePathUsesCharacterCounts documents the Piper contract: its ONNX exports
// carry no duration tensor (VITS durations exist only at training time), so every Piper
// timing is a silence estimate weighted by character count.
func TestPiperEstimatePathUsesCharacterCounts(t *testing.T) {
	const rate = 22050
	boundaries := shared.CharCountBoundaries("alpha beta")
	if len(boundaries) != 2 {
		t.Fatalf("got %d boundaries from %q", len(boundaries), "alpha beta")
	}
	if boundaries[0].TokenCount <= boundaries[1].TokenCount {
		t.Errorf("character counts must weight the prior: %+v", boundaries)
	}

	// alpha (0-0.2 s) | pause (0.2-0.3 s) | beta (0.3-0.5 s) | trailing silence (0.5-1 s)
	samples := make([]float32, rate)
	loud := func(from, to float64) {
		for i := int(from * rate); i < int(to*rate) && i < len(samples); i++ {
			samples[i] = 0.5
		}
	}
	loud(0, 0.2)
	loud(0.3, 0.5)

	got := shared.SilenceBoundaryTimings(samples, rate, boundaries)
	if len(got) != 2 {
		t.Fatalf("got %d timings, want 2", len(got))
	}
	if got[0].End < 0.2 || got[0].End > 0.3 {
		t.Errorf("word boundary at %.3f, want the measured pause (0.200-0.300)", got[0].End)
	}
	if got[1].End < 0.45 || got[1].End > 0.52 {
		t.Errorf("last word ends at %.3f, want the end of the speech near 0.500", got[1].End)
	}
	if got[1].End-got[1].Start <= 0 {
		t.Errorf("second word has no width: %+v", got[1])
	}
	if got[0].Word != "alpha" || got[1].Word != "beta" {
		t.Errorf("word labels lost: %+v", got)
	}
}
