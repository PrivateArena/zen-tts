<!-- codegraph-file-count: 12 -->
# zen-tts

## Purpose
zen-tts is a self-hosted, dual-engine text-to-speech server written in Go (1.24) that exposes an HTTP API plus a terminal UI for switching between the **Kitten** (small ONNX) and **Kokoro** (high-quality ONNX) neural TTS engines. Primary output: synthesized speech playable directly through the server (via `ebitengine/oto`) and/or streamed as 16-bit PCM WAV over HTTP. A bundled Tampermonkey userscript (`tts.user.js`) consumes the API from the browser. Model weights and voice packs are auto-downloaded on first run.

## Architecture
```
[main.go] -> [ui.go RunTUI / tview dashboard] -> [server.go StartServer]
                    |                                  |
                    v                                  v
              [config.go LoadConfig]           [engine.go TTSEngine iface]
                    |                                  |
                    v                       +----------+----------+
            [downloadIfNeeded]             |                     |
                                  [internal/kitten]      [internal/kokoro]
                                          |                     |
                                          +---[internal/shared ONNX/phonemize]
```

## File Tree
```
/
  main.go                 # entrypoint + DebugMsg
  server.go               # HTTP server lifecycle + /tts handler
  engine.go               # TTSEngine/TimingEngine/StreamingEngine ifaces + WAV header
  audio.go                # oto-based local playback + resampler
  config.go               # JSON config, voice registry, model auto-download
  normalizer.go           # text normalization (prosody, phrasing, repeat squashing)
  ui.go                   # tview TUI dashboard + LogMsg
  constants.go            # package-level constants
  tts.user.js             # Tampermonkey userscript consuming /tts
  internal/kitten/        # Kitten ONNX engine implementation
  internal/kokoro/        # Kokoro ONNX engine + voice auto-extract
  internal/shared/        # ONNX runtime init, IPA phonemize, word-timing
```

## Component Roles
### Backend (Go)
| File / Module | Role | LOC | Key Exports (with signatures) |
|---|---|---|---|
| `main.go` | Entrypoint; wires config, TUI, server lifecycle | ~59 | `DebugMsg(format string, v ...interface{})`; `main()` |
| `server.go` | HTTP control plane: thread-safe engine swap, start/stop, `/tts` handler | ~419 | `SwitchEngine(engineType, modelName string) error`; `StartServer(modelName string, port, cpuCore int)`; `StopServer()`; `ToggleServer(model string, port, cpuCore int)`; `ttsHandler(w http.ResponseWriter, r *http.Request)` |
| `engine.go` | Engine contracts + WAV header helpers | ~71 | iface `TTSEngine{ Initialize, Synthesize, SampleRate, Close }`; iface `TimingEngine{ SynthesizeWithTimings }`; iface `StreamingEngine{ SynthesizeStream }`; `writeWavHeader(w io.Writer, sampleRate, dataSize int)`; `makeWavHeader(sampleRate, dataSize int) []byte` |
| `audio.go` | Local audio playback via ebitengine/oto + int16 mono resampler | ~154 | `resampleMonoInt16(input []byte, srcRate, dstRate int) []byte`; `audioPlaybackWorker()`; `playData(data []byte, sampleRate int)`; `playAudio(data []byte, sampleRate int) chan struct{}` |
| `config.go` | Persistent JSON config, voice registry, regex replacements, model downloader, voice sample-rate probe | ~376 | types `EngineConfig`, `ReplacementRule`, `Config`, `VoiceRegistryEntry`, `VoiceRegistry`, `TTSRequest`, `RegexRule`; `LoadConfig()`; `SaveConfig()`; `LoadReplacements()`; `LoadRegistry()`; `getModelPaths(key string) (string, string)`; `downloadIfNeeded(path, url string)`; `getSampleRate(path string) int` |
| `normalizer.go` | Pre-TTS text cleanup: prosodic markers, heuristic phrasing, repeat squashing | ~112 | `normalizeText(text string) string`; `applyProsodicEmphasis(text string) string`; `heuristicPhrasing(text string) string`; `squashRepeats(s string, maxRepeats int) string` |
| `ui.go` | Terminal dashboard (tview): logs, voices, rules screens; thread-safe logger | ~274 | `setupTheme()`; `LogMsg(msg string)`; `buildDashboard() *tview.Flex`; `buildVoices(pages *tview.Pages, actions *tview.List) *tview.Flex`; `buildRules() *tview.Flex`; `RunTUI(port, cpuCore int)` |
| `constants.go` | Compile-time string/numeric constants | ~34 | no exports (skeleton empty) |
| `internal/kitten/engine.go` | Kitten ONNX TTS implementation, NPZ voice loader | ~260 | struct `KittenEngine`; `(*KittenEngine).Initialize(modelPath, configPath string) error`; `(*KittenEngine).Synthesize(text, voice string, speed float32) ([]float32, int, error)`; `(*KittenEngine).SampleRate() int`; `(*KittenEngine).Close() error`; `(*KittenEngine).SynthesizeStream(text, voice string, speed float32, callback func([]float32) bool) error`; `loadNPZVoices(npzPath string) (map[string][]float32, error)` |
| `internal/kokoro/engine.go` | Kokoro ONNX TTS with timings + on-demand voice download/extract | ~363 | struct `KokoroEngine`; `Initialize/Synthesize/SampleRate/Close/SynthesizeStream/SynthesizeWithTimings` (same shape as Kitten, plus `SynthesizeWithTimings` returns `[]shared.WordTiming`); `loadVoiceVector(path string) ([]float32, error)`; `findVoiceMatch(voice string) string`; `downloadAndExtractVoice(voicesDir, voiceName string) error` |
| `internal/shared/shared.go` | Shared ONNX/linguistics: IPA phonemize via espeak-ng piper_phonemize, kagome JP tokenizer, silence-based word timings | ~439 | types `WordTiming`, `WordBoundary`; `TokenizeWordBoundaries(text, voice string) []WordBoundary`; `SilenceBoundaryTimings(samples []float32, sampleRate int, boundaries []WordBoundary) []WordTiming`; `InitONNXRuntime() error`; `Phonemize(text, lang string) ([]string, error)`; `GetLanguageCode(voice string) string`; `Tokenize(text string) []int64`; `ConvertKanjiToHiragana(text string) string`; `TokenizeWithVoice(text, voice string) []int64` |
### Frontend / Browser-side
| File / Module | Role | LOC | Key Exports / Hooks used |
|---|---|---|---|
| `tts.user.js` | Tampermonkey userscript that POSTs selected page text to the local `/tts` endpoint and plays returned WAV via HTMLAudioElement | ~188 | IIFE userscript; fetches `/tts`; `Audio` playback; no module exports |

## Cross-References
| File | Called by / calls | Why it's central |
|---|---|---|
| `server.go` | called by `ui.go` (StartServer/StopServer), `main.go`; calls `SwitchEngine`, `ttsHandler`; fan-in 16+ via `ttsHandler` | Single HTTP front door; all client paths converge here |
| `engine.go` | called by `server.go` (SwitchEngine), `internal/kokoro`, `internal/kitten`; defines all engine interfaces | Contract every backend must satisfy — fan-out hub |
| `internal/shared/shared.go` | called by `internal/kitten` (Initialize/Synthesize), `internal/kokoro` (all 3 paths) | Sole ONNX runtime init + phonemize/timing service for both engines |
| `config.go` | called by `main.go` (`getModelPaths`, `LoadConfig`), `server.go` (`SwitchEngine`); calls `downloadIfNeeded` | Source of truth for models, voices, replacements; auto-provisions assets |
| `ui.go` | called by `main.go` (`RunTUI`, `LogMsg`); called by `ui.go` itself (`buildVoices`); fan-in 19 via `LogMsg` | User-facing control surface; central log broadcaster |
| `internal/kokoro/engine.go` | called by `server.go` (SwitchEngine); calls `downloadAndExtractVoice` (6 refs) | Heavier engine with timing + voice auto-fetch, higher complexity |

## Data Flow
```
[tts.user.js] --POST /tts--> [server.go ttsHandler]
                                    |
                          normalizeText (normalizer.go)
                                    |
                       SwitchEngine -> active TTSEngine iface
                                    |
              +---------------------+---------------------+
              |                                           |
       [internal/kitten]                          [internal/kokoro]
              |                                           |
              +-------> [internal/shared Phonemize/Tokenize/InitONNXRuntime]
                                    |
                          Synthesize* -> []float32 PCM
                                    |
                          resampleMonoInt16 (audio.go) -> oto playback (local)
                                    |
                          writeWavHeader -> WAV bytes -> http.ResponseWriter
```

## Key Architectural Patterns
1. **Interface-segregated engine contract**: `TTSEngine`, `TimingEngine`, `StreamingEngine` in `engine.go` are checked individually at runtime (likely via type assertion in `SwitchEngine`) so Kokoro can expose timings while Kitten stays minimal.
2. **Thread-safe engine swap on the fly**: `SwitchEngine` in `server.go` is documented "safe to call while requests are being processed" — uses atomic/pointer swap so the HTTP handler dispatches against the current engine without restarting.
3. **Auto-provisioning asset layer**: `downloadIfNeeded` (generic) + `downloadAndExtractVoice` (Kokoro-specific ZIP extract) plus `getSampleRate` probe — first-run users get working voices with zero manual setup.
4. **Shared linguistics singleton**: `internal/shared` owns ONNX runtime init (`InitONNXRuntime`) once and exposes `Phonemize`, `Tokenize`, `TokenizeWithVoice`, `ConvertKanjiToHiragana` so engines don't duplicate tokenizer/phonemizer state.
5. **Silence-based word timing**: `SilenceBoundaryTimings` derives per-word timings from amplitude gaps in synthesized PCM — avoids needing model-internal alignment data.
6. **TUI as control plane**: `ui.go` wraps `server.go` lifecycle (Start/Stop/ToggleServer) inside tview pages; `LogMsg` is the single thread-safe sink (mutex-protected) used by every subsystem — fan-in 19.
7. **Normalization pipeline**: `normalizeText` → `applyProsodicEmphasis` → `heuristicPhrasing` → `squashRepeats` forms a composable pre-TTS transform chain in `normalizer.go`.

## Read Triggers
| If you need to... | Open these files |
|---|---|
| Add a new HTTP endpoint | `server.go` (`ttsHandler`), `config.go` (`TTSRequest`) |
| Add a new TTS engine | `engine.go` (interfaces), an existing engine (`internal/kitten/engine.go`) as template, `server.go` (`SwitchEngine`) |
| Change normalization rules | `normalizer.go` |
| Add/modify voice replacements | `config.go` (`LoadReplacements`, `ReplacementRule`, `RegexRule`) |
| Tweak the TUI dashboard | `ui.go` (`buildDashboard`, `buildVoices`, `buildRules`) |
| Change phonemization/tokenization | `internal/shared/shared.go` (`Phonemize`, `Tokenize`, `TokenizeWithVoice`, `ConvertKanjiToHiragana`) |
| Modify auto-download behavior | `config.go` (`downloadIfNeeded`, `getModelPaths`), `internal/kokoro/engine.go` (`downloadAndExtractVoice`) |
| Adjust WAV output format | `engine.go` (`writeWavHeader`, `makeWavHeader`) |
| Tune local audio playback | `audio.go` (`resampleMonoInt16`, `playData`, `playAudio`) |

## Dependencies
### Inference / ML
| Package / Module | Role | Version |
|---|---|---|
| `github.com/yalue/onnxruntime_go` | ONNX Runtime C bindings (single shared init) | v1.11.0 |
| `github.com/ikawaha/kagome/v2` + `kagome-dict/ipa` | Japanese morphological tokenizer + IPA dict | v2.11.0 / v1.2.6 |
### Audio / Playback
| Package / Module | Role | Version |
|---|---|---|
| `github.com/ebitengine/oto/v3` | Low-latency local audio output | v3.4.0 |
### TUI
| Package / Module | Role | Version |
|---|---|---|
| `github.com/rivo/tview` + `gdamore/tcell/v2` | Terminal dashboard framework + cell renderer | v0.42.0 / v2.13.8 |
