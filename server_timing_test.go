package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"zen-tts/internal/shared"
)

// ---------------------------------------------------------------------------
// Test doubles. Each one counts its calls, because "one request, one synthesis"
// is a contract this file has to hold: the double-synthesis bug used to hand back
// audio and timings of two different renderings of the same text.
// ---------------------------------------------------------------------------

type synthCalls struct {
	mu         sync.Mutex
	plain      int
	timed      int
	aligned    int
	withParams int
	lastText   string
	lastVoice  string
	lastSpeed  float32
	lastParams map[string]any
}

func (c *synthCalls) record(kind *int, text, voice string, speed float32, params map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	*kind++
	c.lastText, c.lastVoice, c.lastSpeed, c.lastParams = text, voice, speed, params
}

// fakeEngine is the bare TTSEngine contract: audio only, no timing support at all.
type fakeEngine struct {
	synthCalls
	samples []float32
	rate    int
	err     error
}

func (f *fakeEngine) Initialize(modelPath, configPath string) error { return nil }
func (f *fakeEngine) Close() error                                  { return nil }
func (f *fakeEngine) SampleRate() int {
	if f.rate == 0 {
		return 24000
	}
	return f.rate
}
func (f *fakeEngine) Synthesize(text, voice string, speed float32) ([]float32, int, error) {
	f.record(&f.plain, text, voice, speed, nil)
	if f.err != nil {
		return nil, 0, f.err
	}
	return f.audio(), f.SampleRate(), nil
}
func (f *fakeEngine) audio() []float32 {
	if f.samples != nil {
		return f.samples
	}
	return []float32{0.25, -0.25, 0.5, -0.5}
}

// fakeTimingEngine implements only the older TimingEngine upgrade.
type fakeTimingEngine struct {
	fakeEngine
	words []shared.WordTiming
}

func (f *fakeTimingEngine) SynthesizeWithTimings(text, voice string, speed float32) ([]float32, []shared.WordTiming, int, error) {
	f.record(&f.timed, text, voice, speed, nil)
	if f.err != nil {
		return nil, nil, 0, f.err
	}
	return f.audio(), f.words, f.SampleRate(), nil
}

// fakeAlignedEngine implements the exact-timing upgrade: one pass, words and phonemes.
type fakeAlignedEngine struct {
	fakeEngine
	words    []shared.WordTiming
	phonemes []shared.PhonemeTiming
	source   string
	frameMS  float64
}

func (f *fakeAlignedEngine) SynthesizeAligned(text, voice string, speed float32) (shared.Alignment, error) {
	f.record(&f.aligned, text, voice, speed, nil)
	if f.err != nil {
		return shared.Alignment{}, f.err
	}
	source := f.source
	if source == "" {
		source = shared.TimingSourceDuration
	}
	return shared.Alignment{
		Samples:    f.audio(),
		SampleRate: f.SampleRate(),
		Words:      f.words,
		Phonemes:   f.phonemes,
		Source:     source,
		FrameMS:    f.frameMS,
	}, nil
}

// fakeParamEngine is the Inflect shape: per-request parameters, no timing support.
type fakeParamEngine struct {
	fakeEngine
}

func (f *fakeParamEngine) SynthesizeWithParams(text, voice string, speed float32, params map[string]any) ([]float32, int, error) {
	f.record(&f.withParams, text, voice, speed, params)
	return f.audio(), f.SampleRate(), nil
}

func (f *fakeParamEngine) SynthesizeStreamWithParams(text, voice string, speed float32, params map[string]any, callback func([]float32) bool) error {
	return nil
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

var testWords = []shared.WordTiming{
	{Word: "hello", Start: 0.1, End: 0.45},
	{Word: "world.", Start: 0.5, End: 0.9},
}

var testPhonemes = []shared.PhonemeTiming{
	{Phoneme: "h", WordIndex: 0, Start: 0.1, End: 0.15, Frames: 6},
	{Phoneme: "ə", WordIndex: 0, Start: 0.15, End: 0.45, Frames: 12},
	{Phoneme: "w", WordIndex: 1, Start: 0.5, End: 0.6, Frames: 4},
}

// useEngine installs a fake engine plus a minimal config for the duration of one test.
func useEngine(t *testing.T, synth TTSEngine) {
	t.Helper()

	oldSynth := activeSynth
	oldCfg := CurrentConfig
	CurrentConfig = Config{
		LastModel:    "fake-model",
		Port:         5055,
		ActiveEngine: "fake",
		Engines: map[string]EngineConfig{
			"fake": {Model: "fake-model", Voice: "config-voice", Speed: 1.0},
		},
	}
	activeSynth = synth
	t.Cleanup(func() {
		activeSynth = oldSynth
		CurrentConfig = oldCfg
	})
}

func postTTS(t *testing.T, query string, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/tts"+query, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	ttsHandler(rec, req)
	return rec
}

var errFakeSynth = errors.New("fake synthesis failure")

// timingEnvelope parses the ?timestamps JSON body, reporting problems as errors.
func timingEnvelope(rec *httptest.ResponseRecorder) (map[string]any, error) {
	if rec.Code != http.StatusOK {
		return nil, fmt.Errorf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		return nil, fmt.Errorf("Content-Type = %q, want application/json", ct)
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		return nil, fmt.Errorf("body is not JSON: %v (%.80s)", err, rec.Body.String())
	}
	return out, nil
}

func decodeTimingResponse(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	out, err := timingEnvelope(rec)
	if err != nil {
		t.Fatalf("%v", err)
	}
	return out
}

// ---------------------------------------------------------------------------
// The wire contract
// ---------------------------------------------------------------------------

// TestTimestampsEnvelopeKeepsOldKeys pins the three keys that were already on the wire and
// checks the new ones, so the timing upgrade cannot break existing clients.
func TestTimestampsEnvelopeKeepsOldKeys(t *testing.T) {
	synth := &fakeAlignedEngine{words: testWords, phonemes: testPhonemes, frameMS: 25}
	useEngine(t, synth)

	rec := postTTS(t, "?timestamps=1", `{"text":"hello world","voice":"test-voice","speed":1.0}`)
	out := decodeTimingResponse(t, rec)

	audio, ok := out["audio"].(string)
	if !ok || audio == "" {
		t.Fatalf("audio must stay a non-empty base64 string, got %T", out["audio"])
	}
	wav, err := base64.StdEncoding.DecodeString(audio)
	if err != nil {
		t.Fatalf("audio is not valid base64: %v", err)
	}
	if string(wav[:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		t.Errorf("audio is not a WAV (%q%q)", wav[:4], wav[8:12])
	}
	if got := binary.LittleEndian.Uint32(wav[24:28]); got != 24000 {
		t.Errorf("WAV sample rate = %d, want 24000", got)
	}
	pcm := wav[44:]
	if len(pcm) != 8 {
		t.Errorf("WAV carries %d PCM bytes, want 8 (4 float samples as int16)", len(pcm))
	}
	if got := binary.LittleEndian.Uint32(wav[40:44]); int(got) != len(pcm) {
		t.Errorf("WAV data chunk says %d bytes, actual %d", got, len(pcm))
	}

	timings, ok := out["timings"].([]any)
	if !ok || len(timings) != 2 {
		t.Fatalf("timings must be the two-word array, got %v", out["timings"])
	}
	first, ok := timings[0].(map[string]any)
	if !ok {
		t.Fatalf("timing entry is not an object: %v", timings[0])
	}
	for _, k := range []string{"word", "start", "end"} {
		if _, ok := first[k]; !ok {
			t.Errorf("timing entry lost the %q key that predates this change", k)
		}
	}
	if first["word"] != "hello" || first["start"].(float64) != 0.1 {
		t.Errorf("timing entry changed shape: %v", first)
	}
	if d := out["duration"].(float64); d != float64(len(synth.audio()))/24000.0 {
		t.Errorf("duration = %v, want %v", d, float64(len(synth.audio()))/24000.0)
	}

	// Additive fields.
	if got := out["sample_rate"].(float64); got != 24000 {
		t.Errorf("sample_rate = %v", got)
	}
	if got := out["timing_source"].(string); got != shared.TimingSourceDuration {
		t.Errorf("timing_source = %q, want %q", got, shared.TimingSourceDuration)
	}
	if got := out["frame_ms"].(float64); got != 25 {
		t.Errorf("frame_ms = %v, want the 25 ms model frame", got)
	}
	if _, ok := out["phonemes"]; ok {
		t.Errorf("level 1 must not ship phonemes: %v", out["phonemes"])
	}
}

// TestTimestampsIsOneSynthesis is the regression guard for the double render: the audio and
// the alignment must come out of the same call.
func TestTimestampsIsOneSynthesis(t *testing.T) {
	synth := &fakeAlignedEngine{words: testWords, phonemes: testPhonemes, frameMS: 25}
	useEngine(t, synth)

	postTTS(t, "?timestamps=2", `{"text":"hello world","voice":"test-voice","speed":1.25}`)

	synth.mu.Lock()
	defer synth.mu.Unlock()
	if synth.aligned != 1 {
		t.Errorf("SynthesizeAligned called %d times, want exactly 1", synth.aligned)
	}
	if synth.plain != 0 {
		t.Errorf("Synthesize called %d times for a timed request, want 0", synth.plain)
	}
	if synth.timed != 0 {
		t.Errorf("the deprecated TimingEngine path ran %d times alongside the aligned one", synth.timed)
	}
	// Timings must be computed on the text that was actually synthesized, i.e. after
	// normalization added the sentence-final punctuation.
	if !strings.HasSuffix(synth.lastText, ".") {
		t.Errorf("engine received un-normalized text %q", synth.lastText)
	}
	if synth.lastSpeed != 1.25 || synth.lastVoice != "test-voice" {
		t.Errorf("engine got voice=%q speed=%v", synth.lastVoice, synth.lastSpeed)
	}
}

func TestTimestampsLevelTwoAddsPhonemes(t *testing.T) {
	synth := &fakeAlignedEngine{words: testWords, phonemes: testPhonemes, frameMS: 25}
	useEngine(t, synth)

	out := decodeTimingResponse(t, postTTS(t, "?timestamps=2", `{"text":"hello world","voice":"v"}`))
	phonemes, ok := out["phonemes"].([]any)
	if !ok || len(phonemes) != 3 {
		t.Fatalf("phonemes = %v, want 3 entries", out["phonemes"])
	}
	p := phonemes[2].(map[string]any)
	for _, k := range []string{"phoneme", "word_index", "start", "end", "frames"} {
		if _, ok := p[k]; !ok {
			t.Errorf("phoneme entry lost %q: %v", k, p)
		}
	}
	if p["phoneme"] != "w" || p["word_index"].(float64) != 1 {
		t.Errorf("phoneme entry wrong: %v", p)
	}
}

// TestTimestampsBodyFieldAndQueryOverride covers the JSON-body spelling and precedence.
func TestTimestampsBodyFieldAndQueryOverride(t *testing.T) {
	synth := &fakeAlignedEngine{words: testWords, phonemes: testPhonemes}
	useEngine(t, synth)

	out := decodeTimingResponse(t, postTTS(t, "", `{"text":"hello world","voice":"v","timestamps":2}`))
	if _, ok := out["phonemes"]; !ok {
		t.Errorf("body timestamps=2 must produce phonemes: %v", out)
	}

	out = decodeTimingResponse(t, postTTS(t, "?timestamps=1", `{"text":"hello world","voice":"v","timestamps":2}`))
	if _, ok := out["phonemes"]; ok {
		t.Errorf("the query parameter must win over the body field: %v", out)
	}
}

// TestTimestampsOnLegacyTimingEngine: an engine that predates the alignment API keeps
// working, and is labelled honestly instead of pretending to be exact.
func TestTimestampsOnLegacyTimingEngine(t *testing.T) {
	synth := &fakeTimingEngine{fakeEngine: fakeEngine{}, words: testWords}
	useEngine(t, synth)

	out := decodeTimingResponse(t, postTTS(t, "?timestamps=1", `{"text":"hello world","voice":"v"}`))
	if synth.timed != 1 || synth.plain != 0 {
		t.Errorf("calls: timed=%d plain=%d, want 1/0", synth.timed, synth.plain)
	}
	if got := out["timing_source"].(string); got != shared.TimingSourceUnknown {
		t.Errorf("timing_source = %q, want %q for an engine that cannot say", got, shared.TimingSourceUnknown)
	}
	if len(out["timings"].([]any)) != 2 {
		t.Errorf("timings = %v", out["timings"])
	}
}

// TestTimestampsOnEngineWithoutTimings pins the pre-existing behaviour: audio plus null
// timings, not an error.
func TestTimestampsOnEngineWithoutTimings(t *testing.T) {
	synth := &fakeEngine{}
	useEngine(t, synth)

	out := decodeTimingResponse(t, postTTS(t, "?timestamps=1", `{"text":"hello world","voice":"v"}`))
	raw, _ := json.Marshal(out)
	if !strings.Contains(string(raw), `"timings":null`) {
		t.Errorf("an engine without timings must still yield \"timings\": null, got %s", raw)
	}
	if got := out["timing_source"].(string); got != "none" {
		t.Errorf("timing_source = %q, want \"none\"", got)
	}
	if synth.plain != 1 {
		t.Errorf("plain synthesis calls = %d, want 1", synth.plain)
	}
	if out["audio"].(string) == "" {
		t.Errorf("audio must still be returned")
	}
}

// TestTimestampsKeepsParamsWhenEngineHasNoAlignment guards the parameterized-engine path
// (Inflect) against being replaced by a plain call when timings are requested.
func TestTimestampsKeepsParamsWhenEngineHasNoAlignment(t *testing.T) {
	synth := &fakeParamEngine{}
	useEngine(t, synth)

	postTTS(t, "?timestamps=1", `{"text":"hello world","voice":"v","seed":7,"variation":0.4}`)
	synth.mu.Lock()
	defer synth.mu.Unlock()
	if synth.withParams != 1 || synth.plain != 0 {
		t.Errorf("params must survive the timing path: withParams=%d plain=%d", synth.withParams, synth.plain)
	}
	if got := synth.lastParams["seed"]; got == nil {
		t.Errorf("seed parameter lost: %v", synth.lastParams)
	}
}

// TestPlainRequestStillReturnsWAV pins the default response, which is what the bundled
// userscript and every existing client consumes.
func TestPlainRequestStillReturnsWAV(t *testing.T) {
	synth := &fakeEngine{}
	useEngine(t, synth)

	rec := postTTS(t, "", `{"text":"hello world","voice":"v"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "audio/wav" {
		t.Errorf("Content-Type = %q, want audio/wav", ct)
	}
	body := rec.Body.Bytes()
	if len(body) != 44+8 {
		t.Fatalf("body is %d bytes, want a 44-byte header plus 8 bytes of PCM", len(body))
	}
	if string(body[36:40]) != "data" {
		t.Errorf("header does not carry the data subchunk at its offset: %q", body[36:40])
	}
	if got := binary.LittleEndian.Uint32(body[40:44]); int(got) != 8 {
		t.Errorf("data subchunk size = %d, want 8", got)
	}
	if synth.plain != 1 {
		t.Errorf("plain calls = %d, want 1", synth.plain)
	}
}

func TestTimestampsSynthesisErrorIsNotSilent(t *testing.T) {
	synth := &fakeAlignedEngine{fakeEngine: fakeEngine{err: errFakeSynth}}
	useEngine(t, synth)

	rec := postTTS(t, "?timestamps=1", `{"text":"hello world","voice":"v"}`)
	if rec.Code == http.StatusOK {
		t.Errorf("a failing synthesis must not return a 200 timing envelope: %s", rec.Body.String())
	}
}

// TestConcurrentTimestampedRequests: the HTTP server answers requests in parallel while
// SwitchEngine may be replacing activeSynth, so the timed path must stay race-free.
// Run with -race.
func TestConcurrentTimestampedRequests(t *testing.T) {
	synth := &fakeAlignedEngine{words: testWords, phonemes: testPhonemes, frameMS: 25}
	useEngine(t, synth)

	const n = 8
	var wg sync.WaitGroup
	errCh := make(chan string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errCh <- fmt.Sprintf("panic: %v", r)
				}
			}()
			rec := postTTS(t, "?timestamps=2", `{"text":"hello world","voice":"v"}`)
			out, err := timingEnvelope(rec)
			if err != nil {
				errCh <- err.Error()
				return
			}
			if len(out["timings"].([]any)) != 2 || len(out["phonemes"].([]any)) != 3 {
				errCh <- fmt.Sprintf("bad envelope: %v", out)
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for e := range errCh {
		t.Error(e)
	}
	synth.mu.Lock()
	defer synth.mu.Unlock()
	if synth.aligned != n {
		t.Errorf("aligned calls = %d, want %d (and never %d: one per request)", synth.aligned, n, n*2)
	}
	if synth.plain != 0 {
		t.Errorf("plain calls = %d, want 0", synth.plain)
	}
}

// ---------------------------------------------------------------------------
// Pure request parsing
// ---------------------------------------------------------------------------

func TestTimestampsLevel(t *testing.T) {
	tests := []struct {
		query string
		body  int
		want  int
	}{
		{"", 0, 0},
		{"?timestamps=1", 0, 1},
		{"?timestamps=2", 0, 2},
		{"?timestamps=3", 0, 2}, // clamped: unknown levels degrade to the richest one
		{"?timestamps=0", 2, 0},
		{"?timestamps=true", 0, 1},
		{"?timestamps=false", 0, 0},
		{"?timestamps=nonsense", 0, 0},
		{"?timestamps=", 2, 2}, // empty query falls through to the body
		{"", 1, 1},
		{"", 7, 2},
		{"", -1, 0},
		{"?timestamps=2", 1, 2},
	}
	for _, tc := range tests {
		req := httptest.NewRequest(http.MethodPost, "/tts"+tc.query, nil)
		if got := timestampsLevel(req, &TTSRequest{Timestamps: tc.body}); got != tc.want {
			t.Errorf("query %q body %d -> level %d, want %d", tc.query, tc.body, got, tc.want)
		}
	}
	if got := timestampsLevel(httptest.NewRequest(http.MethodPost, "/tts?timestamps=1", nil), nil); got != 1 {
		t.Errorf("nil request body must not panic, got level %d", got)
	}
}

func TestFrameRateEchoesTheCallersAnimationRate(t *testing.T) {
	tests := []struct {
		query string
		body  float64
		want  float64
	}{
		{"", 0, defaultFrameRate},
		{"?fps=60", 0, 60},
		{"?fps=23.976", 0, 23.976},
		{"?fps=0", 0, defaultFrameRate},
		{"?fps=nope", 0, defaultFrameRate},
		{"", 24, 24},
		{"?fps=30", 24, 30},
		{"?frame_rate=25", 0, 25},
	}
	for _, tc := range tests {
		req := httptest.NewRequest(http.MethodPost, "/tts"+tc.query, nil)
		if got := frameRateOf(req, &TTSRequest{Fps: tc.body}); got != tc.want {
			t.Errorf("query %q body %v -> %v, want %v", tc.query, tc.body, got, tc.want)
		}
	}

	// And it must reach the envelope so a client can do frame = floor(t * frame_rate).
	synth := &fakeAlignedEngine{words: testWords, frameMS: 25}
	useEngine(t, synth)
	out := decodeTimingResponse(t, postTTS(t, "?timestamps=1&fps=60", `{"text":"hello world","voice":"v"}`))
	if got := out["frame_rate"].(float64); got != 60 {
		t.Errorf("frame_rate = %v, want 60", got)
	}
}
