Project: zen-tts
Language: Go (primary), TypeScript (userscript)
Purpose: Multi-engine local text-to-speech system with HTTP API, terminal UI, and browser userscript integration.

Architecture Overview:

zen-tts/
  main.go                   (entry point: CLI/TUI/HTTP bootstrapping)
  engine.go                 (core TTS engine interfaces)
  config.go                 (configuration, voice registry, downloads)
  server.go                 (HTTP server and request routing)
  ui.go                     (terminal user interface)
  audio.go                  (PCM resampling and audio playback)
  normalizer.go             (text normalization, emphasis, phrasing)
  constants.go              (project constants)
  tts.user.js               (browser userscript client)
  internal/
    shared/
      shared.go             (shared ONNX runtime, phonemization, tokenization)
    kokoro/
      engine.go             (Kokoro TTS engine implementation)
    kitten/
      engine.go             (Kitten TTS engine implementation)

Component Roles:

main.go
  - Parses arguments and initializes configuration via LoadConfig
  - Starts HTTP server (StartServer)
  - Launches terminal UI (RunTUI)
  - Coordinates all top-level subsystems

engine.go
  - Defines TTSEngine interface (Initialize, Synthesize, SampleRate, Close)
  - Defines TimingEngine interface (SynthesizeWithTimings)
  - Defines StreamingEngine interface (SynthesizeStream)
  - Exports WAV header helpers (writeWavHeader, makeWavHeader)

config.go
  - EngineConfig, ReplacementRule, Config, VoiceRegistryEntry, VoiceRegistry
  - TTSRequest struct for API payloads
  - LoadConfig / SaveConfig persistence
  - LoadReplacements, LoadRegistry
  - getModelPaths (model discovery)
  - downloadIfNeeded (auto-fetch models)

server.go
  - StartServer / StopServer / ToggleServer
  - SwitchEngine: thread-safe runtime engine swap
  - ttsHandler: main HTTP request handler (highest connectivity file)
  - Bridges HTTP requests to active TTSEngine

ui.go
  - LogMsg: central thread-safe logging
  - buildDashboard, buildVoices, buildRules: TUI screens
  - RunTUI: terminal interface runner

audio.go
  - resampleMonoInt16: sample rate conversion
  - playData / playAudio: playback pipeline
  - audioPlaybackWorker: background audio queue worker

normalizer.go
  - normalizeText
  - applyProsodicEmphasis
  - heuristicPhrasing
  - squashRepeats

internal/shared/shared.go
  - WordTiming, WordBoundary
  - TokenizeWordBoundaries, tokenizeWord
  - SilenceBoundaryTimings
  - InitONNXRuntime
  - Phonemize (espeak-ng / piper_phonemize)
  - GetLanguageCode
  - Tokenize / getJpTokenizer
  - ConvertKanjiToHiragana / katakanaToHiragana
  - TokenizeWithVoice

internal/kokoro/engine.go
  - KokoroEngine struct
  - Implements TTSEngine + StreamingEngine + TimingEngine
  - loadVoiceVector, findVoiceMatch
  - downloadAndExtractVoice

internal/kitten/engine.go
  - KittenEngine struct
  - Implements TTSEngine + StreamingEngine
  - loadNPZVoices

tts.user.js
  - Browser userscript for requesting TTS from local server

Dependency Flow:

main.go
  -> config.go (LoadConfig)
  -> server.go (StartServer)
  -> ui.go (RunTUI)

server.go
  -> config.go (SwitchEngine, downloadIfNeeded)
  -> engine.go (TTSEngine interface)
  -> internal/kokoro/engine.go (Initialize, Synthesize, SynthesizeWithTimings, SynthesizeStream)
  -> internal/kitten/engine.go (Initialize, Synthesize, SynthesizeStream)
  -> audio.go (playData)
  -> ui.go (LogMsg)

config.go
  -> engine.go (Close)

ui.go
  -> config.go (buildVoices)
  -> ui.go (LogMsg)

internal/shared/shared.go
  <- internal/kokoro/engine.go (Initialize, Synthesize, TokenizeWithVoice)
  <- internal/kitten/engine.go (Initialize, Synthesize)

Key Architectural Patterns:

1. Interface-driven engine abstraction
   - TTSEngine, TimingEngine, StreamingEngine in engine.go
   - Both KokoroEngine and KittenEngine implement core interfaces
   - SwitchEngine allows hot-swapping backends at runtime

2. Shared runtime layer
   - internal/shared/centralizes ONNX Runtime bootstrap, phonemization, and tokenization
   - Engines delegate heavy lifting to shared utilities rather than duplicating logic

3. Layered server model
   - HTTP layer (server.go) delegates to engine layer (internal/*/engine.go)
   - Cross-cutting concerns (logging) funnel through ui.go LogMsg

4. Configuration-as-data + download-on-demand
   - Voice registry and replacement rules loaded from disk
   - Models auto-downloaded when missing (downloadIfNeeded, downloadAndExtractVoice)

5. Terminal-first UX with optional browser client
   - Primary interaction via TUI (buildDashboard, buildVoices, buildRules)
   - tts.user.js enables browser pages to consume the local HTTP API

ASCII Dependency Tree (high level):

zen-tts
  main.go
    +- config.go
    |     +- engine.go
    |     +- internal/shared/shared.go
    |
    +- server.go
    |     +- config.go
    |     +- engine.go
    |     |     +- internal/kokoro/engine.go
    |     |     |     +- internal/shared/shared.go
    |     |     +- internal/kitten/engine.go
    |     |           +- internal/shared/shared.go
    |     +- audio.go
    |     +- ui.go
    |
    +- ui.go
          +- config.go
          +- ui.go (internal LogMsg)

  tts.user.js
    +- server.go (HTTP client)

Hotspots / High Connectivity:
  - ui.go LogMsg (degree 19): cross-cutting logger used by multiple subsystems
  - server.go ttsHandler (degree 16): central request processing pipeline
  - config.go downloadIfNeeded (degree 14): shared model acquisition path
  - engine.go Close (degree 14): cleanup across both engine implementations
  - server.go SwitchEngine (degree 10): runtime backend swap coordinator
  - config.go getModelPaths (degree 9): file discovery for voices/model files
  - server.go StartServer (degree 8): HTTP subsystem entry
  - internal/kokoro/engine.go downloadAndExtractVoice (degree 6): Kokoro-specific model handling
  - server.go StopServer (degree 6): HTTP teardown
  - ui.go RunTUI (degree 6): terminal UI entry

Project Root: /media/jang/home/Deve/zen-tts
Generated: 2026-07-05
