package kokoro

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"zen-tts/internal/shared"
)

const (
	kokoroModel  = "models/kokoro/kokoro-v1.0.onnx"
	kokoroConfig = "models/kokoro/kokoro-v1.0.json"

	referenceSentence = "The quick brown fox jumps over the lazy dog, then runs away fast."
	shortSentence     = "Hello there, this is a test."
)

// The Kokoro ONNX audio graph is bit-reproducible, so the waveform of a fixed
// text+voice+speed is a legitimate golden: it proves that routing the tokens through
// BuildTokenStream and reading the style row did not change the audio at all.
var kokoroAudioGoldens = []struct {
	text      string
	samples   int
	pcmSHA256 string
}{
	{
		text:      referenceSentence,
		samples:   121200,
		pcmSHA256: "5879c1ecb1e47efdb5cf504710fd0f0d7c7a371da5bbb1f090a28502bb794c31",
	},
	{
		text:      shortSentence,
		samples:   58200,
		pcmSHA256: "b5668bb0b7c52566a416d9d19026db6b7ce8ce908c1335f2cc5e47a2d95e9804",
	},
}

func pcmSHA256(samples []float32) string {
	h := sha256.New()
	for _, s := range samples {
		pi := int16(s * 32767.0)
		h.Write([]byte{byte(pi), byte(pi >> 8)})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// setupEngine chdirs to the repository root (./piper and ./models are resolved relative
// to it) and initializes the Kokoro engine against the shipped model.
func setupEngine(t *testing.T) *KokoroEngine {
	t.Helper()
	chdirRepoRoot(t)
	for _, p := range []string{kokoroModel} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("%s missing: Kokoro integration test needs the downloaded model", p)
		}
	}
	if err := shared.InitONNXRuntime(); err != nil {
		t.Fatalf("onnxruntime: %v", err)
	}
	e := &KokoroEngine{}
	if err := e.Initialize(kokoroModel, kokoroConfig); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	return e
}

func chdirRepoRoot(t *testing.T) {
	t.Helper()
	orig, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := orig
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			if err := os.Chdir(dir); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { os.Chdir(orig) })
			return
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatal("repository root not found")
}

func TestKokoroAudioIsUnchanged(t *testing.T) {
	e := setupEngine(t)

	for _, g := range kokoroAudioGoldens {
		samples, rate, err := e.Synthesize(g.text, "af_bella", 1.0)
		if err != nil {
			t.Fatalf("%q: %v", g.text, err)
		}
		if rate != 24000 {
			t.Errorf("%q: sample rate %d, want 24000", g.text, rate)
		}
		if len(samples) != g.samples {
			t.Errorf("%q: %d samples, want %d (duration predictor parity broken?)", g.text, len(samples), g.samples)
		}
		if got := pcmSHA256(samples); got != g.pcmSHA256 {
			t.Errorf("%q: PCM hash %s, want %s - the audio contract changed", g.text, got, g.pcmSHA256)
		}
	}
}

// TestKokoroDurationSideCarIsFrameExact is the P1 requirement: with the durations model
// installed, /tts?timestamps=1 must report the model's own alignment, not silence guesses.
func TestKokoroDurationSideCarIsFrameExact(t *testing.T) {
	e := setupEngine(t)
	if e.durationsPath == "" {
		t.Skipf("duration side-car %s missing: word timings legitimately fall back to silence analysis",
			DurationsModelPath(kokoroModel))
	}

	a, err := e.SynthesizeAligned(referenceSentence, "af_bella", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if a.Source != shared.TimingSourceDuration {
		t.Errorf("source = %q, want %q", a.Source, shared.TimingSourceDuration)
	}
	if a.FrameMS != 25.0 {
		t.Errorf("frame_ms = %v, want 25 (600 samples/frame at 24 kHz)", a.FrameMS)
	}
	if len(a.Words) != 13 {
		t.Fatalf("got %d words, want 13: %v", len(a.Words), a.Words)
	}

	// sum(durations)*600 == 121200 == len(audio): the whole utterance is accounted for.
	if last := a.Words[len(a.Words)-1].End; last < a.Duration()-0.15 || last > a.Duration()+1e-9 {
		t.Errorf("last word ends at %.4f, want the end of the %.4f s utterance", last, a.Duration())
	}
	// Frame-exact: every boundary is a multiple of one model frame.
	for i, w := range a.Words {
		if mathMod(w.Start, a.FrameMS/1000) > 1e-9 && mathMod(a.Duration()-w.Start, a.FrameMS/1000) > 1e-9 {
			t.Errorf("word %d start %.6f is not on a %v ms frame", i, w.Start, a.FrameMS)
		}
	}
	// Reference alignment measured against the production audio (scratch/timing/kokoro_duration_parity.json).
	wantFirst, wantLast := 0.425, 5.025
	if diff := a.Words[0].Start - wantFirst; diff > 0.026 || diff < -0.026 {
		t.Errorf("first word starts at %.3f, want %.3f (BOS frame)", a.Words[0].Start, wantFirst)
	}
	if diff := a.Words[len(a.Words)-1].End - wantLast; diff > 0.026 || diff < -0.026 {
		t.Errorf("last word ends at %.3f, want %.3f", a.Words[len(a.Words)-1].End, wantLast)
	}
}

func TestKokoroAlignedSingleSynthesisAndPhonemes(t *testing.T) {
	e := setupEngine(t)

	a, err := e.SynthesizeAligned(shortSentence, "af_bella", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if pcmSHA256(a.Samples) != kokoroAudioGoldens[1].pcmSHA256 {
		t.Errorf("SynthesizeAligned must return the very audio Synthesize returns")
	}
	if want := len(wordsOf(shortSentence)); len(a.Words) != want {
		t.Errorf("got %d words for %q, want %d", len(a.Words), shortSentence, want)
	}
	for i, w := range a.Words {
		if w.End <= w.Start {
			t.Errorf("word %d (%q) has no width: %+v", i, w.Word, w)
		}
		if i > 0 && w.Start < a.Words[i-1].End {
			t.Errorf("word %d (%q) overlaps its predecessor", i, w.Word)
		}
	}
	if len(a.Phonemes) == 0 {
		t.Logf("no phonemes (side-car absent): word timings only, source %q", a.Source)
	}
	for _, p := range a.Phonemes {
		if p.End <= p.Start {
			t.Errorf("zero-width phoneme %q: %+v", p.Phoneme, p)
		}
		if p.WordIndex < 0 || p.WordIndex >= len(a.Words) {
			t.Errorf("phoneme %q is attributed to word %d of %d", p.Phoneme, p.WordIndex, len(a.Words))
		}
	}
}

// TestKokoroLegacyTimingsEngineStillWorks pins the TimingEngine contract that predates the
// alignment API: audio plus per-word timings, from a single pass.
func TestKokoroLegacyTimingsEngineStillWorks(t *testing.T) {
	e := setupEngine(t)

	samples, timings, rate, err := e.SynthesizeWithTimings(shortSentence, "af_bella", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 24000 {
		t.Errorf("rate %d", rate)
	}
	if pcmSHA256(samples) != kokoroAudioGoldens[1].pcmSHA256 {
		t.Errorf("legacy timed path returned different audio")
	}
	if len(timings) != 6 {
		t.Errorf("got %d word timings for %q, want 6", len(timings), shortSentence)
	}
	for i, w := range timings {
		if w.End <= w.Start {
			t.Errorf("timing %d has no width: %+v", i, w)
		}
	}
}

// TestKokoroTimingsWithoutSideCar: without the duration model the engine must still answer
// with well-formed estimates and say so - never with silence-derived numbers labelled as
// exact, and never by failing the request.
func TestKokoroTimingsWithoutSideCar(t *testing.T) {
	e := setupEngine(t)
	e.durationsPath = ""

	a, err := e.SynthesizeAligned(referenceSentence, "af_bella", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if a.Source != shared.TimingSourceVAD {
		t.Errorf("source = %q, want %q", a.Source, shared.TimingSourceVAD)
	}
	if a.FrameMS != float64(shared.VADWindowMS) {
		t.Errorf("frame_ms = %v, want the %d ms analysis window of an estimate", a.FrameMS, shared.VADWindowMS)
	}
	if len(a.Words) != 13 {
		t.Fatalf("got %d words, want 13: %v", len(a.Words), a.Words)
	}
	if len(a.Phonemes) != 0 {
		t.Errorf("phoneme timings are not derivable from silence analysis, got %+v", a.Phonemes)
	}
	duration := a.Duration()
	for i, w := range a.Words {
		if w.End <= w.Start {
			t.Errorf("word %d (%q) has no width: %+v", i, w.Word, w)
		}
		if w.End > duration+1e-9 || w.Start < 0 {
			t.Errorf("word %d (%q) escapes the audio: %+v of %.4f s", i, w.Word, w, duration)
		}
		if i > 0 && w.Start < a.Words[i-1].End {
			t.Errorf("word %d (%q) overlaps its predecessor", i, w.Word)
		}
	}
	// The leading silence must not be eaten by the first word - the defect that put
	// "The" at 0.000 while the audio did not start until 0.390.
	if a.Words[0].Start < 0.2 {
		t.Errorf("first word starts at %.3f, want the end of the leading silence", a.Words[0].Start)
	}
}

func TestKokoroDurationsModelPath(t *testing.T) {
	if got := DurationsModelPath("models/kokoro/kokoro-v1.0.onnx"); got != "models/kokoro/kokoro-v1.0-durations.onnx" {
		t.Errorf("DurationsModelPath = %q", got)
	}
	e := setupEngine(t)
	if e.durationsPath != DurationsModelPath(kokoroModel) && e.durationsPath != "" {
		t.Errorf("resolved side-car = %q", e.durationsPath)
	}
}

func wordsOf(text string) []string {
	out := make([]string, 0, 8)
	start := -1
	for i := 0; i <= len(text); i++ {
		if i == len(text) || text[i] == ' ' {
			if start >= 0 {
				out = append(out, text[start:i])
				start = -1
			}
			continue
		}
		if start < 0 {
			start = i
		}
	}
	return out
}

// mathMod reports how far x is from the nearest multiple of m.
func mathMod(x, m float64) float64 {
	if m <= 0 {
		return 0
	}
	q := x / m
	diff := q - float64(int64(q))
	if diff < 0 {
		diff = -diff
	}
	if diff > 0.5 {
		diff = 1 - diff
	}
	return diff * m
}
