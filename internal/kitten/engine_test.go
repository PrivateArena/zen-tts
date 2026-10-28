package kitten

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"zen-tts/internal/shared"
)

const (
	kittenModel  = "models/kitten/kitten-tts-mini.onnx"
	kittenConfig = "models/kitten/kitten-tts-mini.json"
	kittenVoices = "models/kitten/voices.npz"

	kittenReference = "The quick brown fox jumps over the lazy dog, then runs away fast."
)

// setupEngine chdirs to the repository root (the ONNX runtime and the phonemizer are
// resolved relative to it), then loads the shipped KittenTTS mini model. Packages whose
// model files are not present are skipped rather than failed.
func setupEngine(t *testing.T) *KittenEngine {
	t.Helper()
	chdirRepoRoot(t)
	for _, p := range []string{kittenModel, kittenVoices} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("%s missing: KittenTTS integration test needs the downloaded model", p)
		}
	}
	if err := shared.InitONNXRuntime(); err != nil {
		t.Fatalf("onnxruntime: %v", err)
	}
	e := &KittenEngine{}
	if err := e.Initialize(kittenModel, kittenConfig); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	if len(e.voices) == 0 {
		t.Fatal("no voices loaded")
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

// TestKittenDurationTensorIsFrameExact is the P0 contract: the graph's own duration
// output accounts for the waveform sample for sample, which is what makes the reported
// timings exact rather than estimated.
func TestKittenDurationTensorIsFrameExact(t *testing.T) {
	e := setupEngine(t)

	for _, tc := range []struct {
		text    string
		samples int
	}{
		{kittenReference, 136800},
		{"Hello there, this is a test.", 76800},
	} {
		a, err := e.SynthesizeAligned(tc.text, "luna", 1.0)
		if err != nil {
			t.Fatalf("%q: %v", tc.text, err)
		}
		if a.Source != shared.TimingSourceDuration {
			t.Errorf("%q: timing source = %q, want %q (the shipped graph emits 'duration')",
				tc.text, a.Source, shared.TimingSourceDuration)
		}
		if len(a.Samples) != tc.samples {
			t.Errorf("%q: %d samples, want %d (audio path regression)", tc.text, len(a.Samples), tc.samples)
		}
		if a.SampleRate != 24000 {
			t.Errorf("%q: sample rate %d, want 24000", tc.text, a.SampleRate)
		}
		if a.FrameMS != 25.0 {
			t.Errorf("%q: frame resolution %v ms, want 25 ms (600 samples at 24 kHz)", tc.text, a.FrameMS)
		}
		// sum(duration)*600 == len(waveform) exactly, so the only audio after the last
		// word is the EOS frame the model renders as a closing pause.
		if len(a.Words) == 0 {
			t.Fatalf("%q: no word timings", tc.text)
		}
		last := a.Words[len(a.Words)-1]
		d := a.Duration()
		if last.End > d+1e-9 {
			t.Errorf("%q: last word ends at %.4f, past the audio (%.4f)", tc.text, last.End, d)
		}
		if last.End < d-0.15 {
			t.Errorf("%q: alignment stops at %.4f of a %.4f s utterance", tc.text, last.End, d)
		}
		// Phonemes cover everything the words claim, and nothing more.
		for _, p := range a.Phonemes {
			if p.End > last.End+1e-9 {
				t.Errorf("%q: phoneme %q ends (%.4f) after the last word (%.4f)", tc.text, p.Phoneme, p.End, last.End)
			}
		}
	}
}

// TestKittenWordTimings pins the alignment of the reference sentence. They are model
// frames, so they are stable across runs even though the waveform itself is not.
func TestKittenWordTimings(t *testing.T) {
	e := setupEngine(t)

	a, err := e.SynthesizeAligned(kittenReference, "luna", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		word       string
		start, end float64
	}{
		{"The", 0.475, 0.575},
		{"quick", 0.650, 0.925},
		{"brown", 0.975, 1.275},
		{"fox", 1.325, 1.725},
		{"jumps", 1.800, 2.175},
		{"over", 2.225, 2.450},
		{"the", 2.500, 2.600},
		{"lazy", 2.650, 2.950},
		{"dog,", 3.025, 3.775},
		{"then", 3.825, 4.025},
		{"runs", 4.100, 4.350},
		{"away", 4.400, 4.625},
		{"fast.", 4.700, 5.625},
	}
	if len(a.Words) != len(want) {
		t.Fatalf("got %d words, want %d: %v", len(a.Words), len(want), a.Words)
	}
	for i, w := range want {
		got := a.Words[i]
		if got.Word != w.word {
			t.Errorf("word %d = %q, want %q", i, got.Word, w.word)
		}
		if abs(got.Start-w.start) > 0.026 || abs(got.End-w.end) > 0.026 {
			t.Errorf("word %d (%q) = %.3f->%.3f, want %.3f->%.3f", i, w.word, got.Start, got.End, w.start, w.end)
		}
	}
	if a.Words[0].Start < 0.3 {
		t.Errorf("first word starts at %.3f inside the leading silence", a.Words[0].Start)
	}
}

func TestKittenPhonemeTimings(t *testing.T) {
	e := setupEngine(t)

	a, err := e.SynthesizeAligned(kittenReference, "luna", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Phonemes) == 0 {
		t.Fatal("no phoneme timings: the duration tensor is per token, so these are free")
	}
	if len(a.Words) == 0 {
		t.Fatal("no word timings")
	}
	for _, p := range a.Phonemes {
		if strings.TrimSpace(p.Phoneme) == "" {
			t.Errorf("blank phoneme label: %+v", p)
		}
		if p.End <= p.Start {
			t.Errorf("zero-width phoneme %q: %+v", p.Phoneme, p)
		}
		if p.WordIndex < 0 || p.WordIndex >= len(a.Words) {
			t.Errorf("phoneme %q attributed to word %d of %d", p.Phoneme, p.WordIndex, len(a.Words))
			continue
		}
		w := a.Words[p.WordIndex]
		if p.Start < w.Start-1e-9 || p.End > w.End+1e-9 {
			t.Errorf("phoneme %q (%.3f-%.3f) escapes its word %q (%.3f-%.3f)",
				p.Phoneme, p.Start, p.End, w.Word, w.Start, w.End)
		}
	}
	// Phonemes must be emitted in time order so a client can walk them once.
	for i := 1; i < len(a.Phonemes); i++ {
		if a.Phonemes[i].Start < a.Phonemes[i-1].Start {
			t.Fatalf("phonemes out of order at %d: %+v then %+v", i, a.Phonemes[i-1], a.Phonemes[i])
		}
	}
}

func TestKittenLegacyTimingsAgreeWithAlignment(t *testing.T) {
	e := setupEngine(t)

	samples, timings, rate, err := e.SynthesizeWithTimings(kittenReference, "luna", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 24000 {
		t.Errorf("rate %d", rate)
	}
	if len(timings) != 13 {
		t.Fatalf("got %d timings, want 13", len(timings))
	}
	if len(samples) == 0 {
		t.Fatal("SynthesizeWithTimings must return the audio as well, not re-synthesize it")
	}
	plain, _, err := e.Synthesize(kittenReference, "luna", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(plain) != len(samples) {
		t.Errorf("timed call returned %d samples, plain call %d", len(samples), len(plain))
	}
}

func TestKittenSpeedScalesAudioAndTimingsTogether(t *testing.T) {
	e := setupEngine(t)

	slow, err := e.SynthesizeAligned(kittenReference, "luna", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	fast, err := e.SynthesizeAligned(kittenReference, "luna", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(fast.Samples) >= len(slow.Samples) {
		t.Errorf("speed 1.5 produced %d samples against speed 1.0 %d", len(fast.Samples), len(slow.Samples))
	}
	if len(fast.Words) != len(slow.Words) {
		t.Fatalf("word count changed with speed: %d vs %d", len(fast.Words), len(slow.Words))
	}
	for i, w := range fast.Words {
		if w.End > fast.Duration()+1e-9 {
			t.Errorf("speed 1.5 word %d ends at %.3f past %.3f", i, w.End, fast.Duration())
		}
		if w.End-slow.Words[i].End > 1e-6 {
			t.Errorf("word %d (%q) ends later at speed 1.5 (%.3f) than at speed 1.0 (%.3f)",
				i, w.Word, w.End, slow.Words[i].End)
		}
	}
}

func TestKittenUnknownVoiceStillAligns(t *testing.T) {
	e := setupEngine(t)

	a, err := e.SynthesizeAligned("One, two, three.", "nosuchvoice", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Words) != 3 {
		t.Errorf("got %d words, want 3", len(a.Words))
	}
	if a.Source != shared.TimingSourceDuration {
		t.Errorf("source %q", a.Source)
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
