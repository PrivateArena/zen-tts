package main

import (
	"bytes"
	"testing"

	"zen-tts/internal/inflect"
	"zen-tts/internal/kitten"
	"zen-tts/internal/kokoro"
	"zen-tts/internal/piper"
)

// The engine matrix of the timing API, as a compile-time contract: if an engine ever
// stops implementing its alignment methods, /tts?timestamps silently drops back to
// "timings": null for that engine, and nothing else would notice.
var (
	_ TTSEngine = (*piper.PiperEngine)(nil)
	_ TTSEngine = (*kokoro.KokoroEngine)(nil)
	_ TTSEngine = (*kitten.KittenEngine)(nil)
	_ TTSEngine = (*inflect.InflectEngine)(nil)

	// Frame-exact alignment where the model exposes durations.
	_ ExactTimingEngine = (*kitten.KittenEngine)(nil)
	_ ExactTimingEngine = (*kokoro.KokoroEngine)(nil)

	// Silence-derived word timings for engines whose graph has no duration output.
	_ TimingEngine = (*piper.PiperEngine)(nil)
	_ TimingEngine = (*kitten.KittenEngine)(nil)
	_ TimingEngine = (*kokoro.KokoroEngine)(nil)

	_ StreamingEngine = (*piper.PiperEngine)(nil)
	_ StreamingEngine = (*kitten.KittenEngine)(nil)
	_ StreamingEngine = (*kokoro.KokoroEngine)(nil)

	_ ParameterizedEngine = (*inflect.InflectEngine)(nil)
)

// TestTimingEnginesAreDiscoveredThroughAssertions documents what the handler's type
// switches resolve for each engine, which is what decides the timing_source of a response.
func TestTimingEngineSupportMatrix(t *testing.T) {
	tests := []struct {
		name       string
		engine     TTSEngine
		wantExact  bool
		wantTiming bool
	}{
		{"kitten", &kitten.KittenEngine{}, true, true},
		{"kokoro", &kokoro.KokoroEngine{}, true, true},
		{"piper", &piper.PiperEngine{}, true, true},
		{"inflect", &inflect.InflectEngine{}, false, false},
	}
	for _, tc := range tests {
		if _, ok := tc.engine.(ExactTimingEngine); ok != tc.wantExact {
			t.Errorf("%s: ExactTimingEngine = %v, want %v", tc.name, ok, tc.wantExact)
		}
		if _, ok := tc.engine.(TimingEngine); ok != tc.wantTiming {
			t.Errorf("%s: TimingEngine = %v, want %v", tc.name, ok, tc.wantTiming)
		}
	}
}

// TestMakeWavHeaderMatchesStreamedHeader keeps the two header paths in sync: the
// ?timestamps envelope embeds the same bytes the plain response streams.
func TestMakeWavHeaderMatchesStreamedHeader(t *testing.T) {
	var buf bytes.Buffer
	writeWavHeader(&buf, 24000, 8)
	if got, want := buf.String(), string(makeWavHeader(24000, 8)); got != want {
		t.Errorf("makeWavHeader and writeWavHeader disagree")
	}
	if len(makeWavHeader(22050, 0)) != 44 {
		t.Errorf("WAV header must be 44 bytes")
	}
}
