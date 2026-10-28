package shared

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// chdirRepoRoot puts the test binary where the production code expects to run: the
// phonemizer and the ONNX runtime are looked up through "./piper".
func chdirRepoRoot(t *testing.T) {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	orig := dir
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "piper", "piper_phonemize")); err == nil {
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
	t.Skip("piper_phonemize not found; phonemization tests need the real espeak-ng bundle")
}

const referenceSentence = "The quick brown fox jumps over the lazy dog, then runs away fast."

// referenceTokens is the ONNX ID sequence the pre-timing implementation fed to the Kokoro
// audio graph for referenceSentence (121,200 samples at 24 kHz). Everything added on top
// of it - word grouping, durations, alignment - must not move it by one token.
var referenceTokens = []int64{
	0, 81, 83, 16, 53, 65, 156, 102, 53, 16, 44, 123, 156, 43, 135, 56, 16, 48, 156, 69,
	158, 53, 61, 16, 46, 147, 156, 138, 55, 58, 61, 16, 157, 57, 135, 64, 85, 16, 81, 83,
	16, 54, 156, 47, 102, 68, 51, 16, 46, 156, 69, 158, 92, 3, 16, 81, 156, 86, 56, 16,
	123, 156, 138, 56, 68, 16, 70, 65, 156, 47, 102, 16, 48, 156, 72, 61, 62, 4, 0,
}

func TestTokenizeWithVoiceIsUnchanged(t *testing.T) {
	chdirRepoRoot(t)

	if got := TokenizeWithVoice(referenceSentence, "af_bella"); !equalIDs(got, referenceTokens) {
		t.Errorf("token stream changed:\n got %v\nwant %v", got, referenceTokens)
	}
	if got := Tokenize("Hello there, this is a test."); len(got) == 0 {
		t.Errorf("Tokenize produced no tokens")
	}
	// Tokenize is the en-us view of the same stream.
	if a, b := Tokenize(referenceSentence), TokenizeWithVoice(referenceSentence, "en-us"); !equalIDs(a, b) {
		t.Errorf("Tokenize and TokenizeWithVoice(en-us) disagree")
	}
}

func TestBuildTokenStreamMatchesPlainTokenizer(t *testing.T) {
	chdirRepoRoot(t)

	for _, text := range []string{
		referenceSentence,
		"Hello there, this is a test.",
		"One, two, three, four, five, six, seven.",
		"  uneven   spacing  ",
		"こんにちは",
	} {
		ts := BuildTokenStream(text, "af_bella")
		if got := TokenizeWithVoice(text, "af_bella"); !equalIDs(got, ts.IDs) {
			t.Errorf("text %q: TokenizeWithVoice diverged from BuildTokenStream", text)
		}
		if len(ts.Labels) != len(ts.IDs) {
			t.Errorf("text %q: %d labels for %d tokens", text, len(ts.Labels), len(ts.IDs))
		}
	}

	ts := BuildTokenStream(referenceSentence, "af_bella")
	if !equalIDs(ts.IDs, referenceTokens) {
		t.Errorf("reference token stream drifted: %v", ts.IDs)
	}
}

// TestBuildTokenStreamGroupsMatchTheSynthesizedTokens is the fix for the 66-vs-77 defect:
// the per-word token counts used as the timing prior must be the counts the model actually
// synthesized, obtained from ONE phonemization pass.
func TestBuildTokenStreamGroupsMatchTheSynthesizedTokens(t *testing.T) {
	chdirRepoRoot(t)

	ts := BuildTokenStream(referenceSentence, "af_bella")
	wantWords := strings.Fields(referenceSentence)
	if !ts.WordsExact {
		t.Fatalf("13 source words should label 13 phoneme groups, got %v", ts.Words)
	}
	if len(ts.Words) != len(wantWords) {
		t.Fatalf("got %d groups for %d words", len(ts.Words), len(wantWords))
	}
	for i, w := range ts.Words {
		if w.Word != wantWords[i] {
			t.Errorf("group %d labelled %q, want %q", i, w.Word, wantWords[i])
		}
	}

	boundaries, ok := ts.WordBoundaries()
	if !ok {
		t.Errorf("WordBoundaries refused an exactly-labelled stream")
	}
	spaces := 0
	for _, id := range ts.IDs[1 : len(ts.IDs)-1] {
		if id == SpaceTokenID {
			spaces++
		}
	}
	bodyTokens := len(ts.IDs) - 2
	total := 0
	for _, b := range boundaries {
		total += b.TokenCount
	}
	if total != bodyTokens-spaces {
		t.Errorf("per-word counts sum to %d, want %d (%d body tokens - %d spaces)", total, bodyTokens-spaces, bodyTokens, spaces)
	}
}

func equalIDs(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
