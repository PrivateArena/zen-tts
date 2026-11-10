# Competitive Intelligence: zen-tts vs. `audio.cpp` (+ local-first TTS adjacency)

**Question asked:** *Do we keep developing zen-tts now that https://github.com/0xShug0/audio.cpp exists?*
**Answer: YES — continue, but not for the reasons we assumed.** Our headline moat is shallower than believed, our headline performance weakness is a myth, and the surviving differentiator is narrower and more specific than "timing". Details in §1 and §6.

Date of study: repo HEAD `0382d37` (audio.cpp v0.9.4, 2026-11-10) measured on the same host as our working tree.

---

## 1. Executive Summary & Our Specialization

**What zen-tts is** (per `PROJECT_OVERVIEW.md` + `TTS_API.md`): a self-hosted, single-binary Go (1.24) TTS server with a `tview` TUI control plane, running four local ONNX engines behind one interface contract (`TTSEngine` / `TimingEngine` / `StreamingEngine` in `engine.go`), with hot engine/voice swap, auto-provisioned weights, local `oto` playback, a browser userscript client, and — the strategic asset — a **timing envelope returned from the same synthesis pass as the audio** (`?timestamps=1|2` → `timings[]`, `phonemes[]`, `duration`, `sample_rate`, `frame_rate`, `frame_ms`, `timing_source`).

**What is genuinely special (verified this session):**

| # | Differentiator | Why it is architectural, not cosmetic |
|---|---|---|
| D1 | **Timing emitted with audio, one request, one pass** | Word/phoneme boundaries are cumsum'd from the *duration tensor* of the model that produced the waveform (`internal/shared/alignment.go`, `duration_graph`). Audio and alignment cannot describe different renders. Competitors either have no timing, or buy it with a **second ASR pass** (forced aligner) that costs an extra round-trip and inherits ASR error. |
| D2 | **Frame-domain contract for animation clients** | `frame_rate` echo + `frame_ms` resolution + `timing_source` provenance label (`duration_graph` / `vad_estimate` / `estimated` / `none`) let hyperframes/zen-board map `frame_index = floor(t × fps)` and know the *error budget* of each number. No competitor exposes an equivalent self-describing envelope. |
| D3 | **Multi-engine honesty matrix** | Four engines behind one HTTP contract with a documented per-engine timing-capability matrix, including graceful degradation to VAD estimate when the graph has no duration output. |
| D4 | **Operator surface** | Go single binary (25 MB), 4.9 s clean build, TUI with live log/voice/rule screens, runtime engine swap under load. audio.cpp ships a WebUI instead of a TUI and a 482 MB build tree. |
| D5 | **Pre-TTS text normalizer + user regex replacement rules** (`normalizer.go`, `config.json`) as a first-class, editable, product-level feature for read-aloud quality. |

**What is NOT special (corrected this session — see §5):**
- ~~"CPU inference is our advantage"~~ → throughput is ~equal; our *latency floor and memory* are worse than they should be, and the cause is our own unimplemented plumbing (§5).
- ~~"audio.cpp structurally cannot do frame-locked timing"~~ → they already compute it inside their Kokoro session and already serve audio+`words` in one JSON body on `/v1/tasks/run`. Closing the last gap for them is a handler change, not a port (§6.1).
- "Per-word timing is exact" → for Kokoro our `word` strings are **eSpeak phoneme groups**, the same defect we would cite against them (§6.2).

---

## 2. Competitor Landscape

| Product | Stack | Scale | Target persona | Timing from TTS? |
|---|---|---|---|---|
| **audio.cpp** (primary threat) | C++ on vendored `ggml` (25 MB src), GGUF weights, cpp-httplib server, embedded native WebUI; CPU/CUDA/Vulkan/HIP/Metal/SYCL/RKNN NPU/WASM+WebGPU | **143 model families / 250+ variants**, 38 with voice cloning; 4,150 ★ / 502 forks; ~1 release/week; Apache-2.0, ShugoAI LLC; powers Unsloth Studio | Local-AI power users, GPU owners, "one runtime for all audio models", app builders wanting OpenAI-compatible `/v1/audio/speech` | Kokoro only, opt-in `return_timestamps`, **phoneme groups**, `confidence` hardcoded 0, exposed on task/CLI routes, not on `/v1/audio/speech` |
| **sherpa-onnx** (k2-fsa) | C++/ONNX Runtime, CPU/embedded-first, `provider="cpu"`, 48+ TTS languages | 15,862 ★, Apache-2.0; VITS/Matcha/Kokoro/Kitten/ZipVoice/Pocket/Supertonic | Embedded/edge, mobile SDK users, multilingual | **None** — offline TTS struct returns samples + sample rate only; timestamps exist only for ASR |
| **Piper → OHF-Voice/piper1-gpl** | C++/ONNX + eSpeak-ng, Python HTTP server | 6,237 ★ (GPL-3.0), "looking for maintainers"; 44 locales | Home Assistant / embedded voice assistants | **Experimental**: per-phoneme `alignments` require `patch_voice_with_alignment`, and leak via a global `/info → last` side channel, not per request |
| **Kokoro-FastAPI** (remsky) | Python + kokoro-onnx/Misaki, Docker multi-arch, OpenAI-compatible + ElevenLabs/Cartesia WebSocket emulation | 5,664 ★, Apache-2.0, 8 languages | Self-hosters wanting an OpenAI/ElevenLabs drop-in | Word-level via `/dev/captioned_speech` (JSON + streamed); chunk timings written to a **sidecar file** (`X-Timing-Path`), explicitly "not word-level" |
| **HeadTTS** (met4citizen) | Browser WASM/WebGPU + Node.js WS/REST | 184 ★, MIT | Realtime avatar/webapp devs | **Yes** — Kokoro `-timestamped` ONNX returns `words/wtimes, visemes/vtimes, phonemes` with audio in one message. Closest analogue to D1; Kokoro-only, Node/WebGPU |
| **Commercial reference** | ElevenLabs `.../with-timestamps` (character-level) + separate forced-alignment; Azure `WordBoundary`/bookmark/viseme (en-US); Polly speech marks `sentence\|word\|viseme\|ssml`; Google only SSML `<mark>` timepoints | — | SaaS | Word/char timing is **table stakes**; phoneme/viseme timing exists **only in cloud APIs** |

**Read of the board.** Word timing is commoditised; *phoneme-level, frame-mapped, provenance-labelled timing from a local process, in the same response as the audio* is delivered today only by zen-tts and HeadTTS — and HeadTTS is a Kokoro-only browser/Node runtime. That is the defensible niche. audio.cpp competes on breadth, GPU throughput and ecosystem velocity; it does not compete on our contract, and its alignment story is a *separate* ASR model (`qwen3_forced_aligner`, `mms_forced_aligner`, task `align`) plus a `/v1/audio/alignments` endpoint.

---

## 3. Feature Comparison Matrix

Legend: **Ours** = zen-tts today · ✔ comparable/better · **STRONG** = our differentiator · ~ = present but weaker · ✖ = missing · **N/A** = deliberately out of scope.

### 3.1 Our Core Advantage / Differentiator
| Capability | Ours | audio.cpp | sherpa-onnx | Piper1-gpl | Kokoro-FastAPI | Commercial |
|---|---|---|---|---|---|---|
| Word+**phoneme** timing in the *same* response as audio, one synthesis pass | **STRONG** (`?timestamps=2`, `duration_graph`) | ~ Kokoro-only, groups, not on `/v1/audio/speech` | ✖ | ~ experimental, `/info` side channel | ✖ word only | char/phoneme via cloud |
| Frame-domain contract (`frame_rate`, `frame_ms`) | **STRONG** | ✖ | ✖ | ✖ | ✖ | ✖ (ms only) |
| Timing **provenance** per engine (`timing_source`) + honest degradation | **STRONG** | ✖ (`confidence`=0 hardcoded) | ✖ | ✖ | ✖ | ✖ |
| Bit-reproducible Kokoro audio+timing pair, documented determinism matrix | **STRONG** (TTS_API §Determinism) | ✖ unverified | ✖ | ✖ | ✖ | ✖ |
| Hot engine swap under load + TUI control plane | **STRONG** | ~ load/unload API, WebUI, no TUI (NOT FOUND) | ✖ (library) | ✖ | ✖ | ✖ |
| Single static-ish Go binary + editable regex normalizer rules | ✔ | ~ composite builds but 482 MB tree | ✔ libs | ✔ | ✖ Docker/Python | N/A |

### 3.2 We Have It (comparable)
| Capability | Ours | audio.cpp | Notes |
|---|---|---|---|
| OpenAI-ish HTTP TTS endpoint | ✔ `/tts` | ✔ `/v1/audio/speech` (+ `/v1/models`, `/health`) | They win on ecosystem drop-in compatibility (§5.7) |
| WAV 16-bit PCM mono, base64 JSON envelope | ✔ | ✔ (`response_format=json\|b64_json`) | parity |
| Voice speed control | ✔ | ✔ (`speed`, presets) | parity |
| Multi-engine behind one contract | ✔ 4 engines | ✔ 143 families | same pattern, different scale |
| Per-request voice selection | ✔ | ✔ + base64 `voice_ref` ≤5 MB | they are richer |
| ONNX-free native phonemizer (eSpeak-ng) | ✔ via `piper_phonemize` subprocess | ✔ shared `espeak_phonemizer`, optional static | **same upstream defect source** (§6.2) |
| Japanese text handling (kagome + kanji→kana) | ✔ | ✔ (model-dependent) | narrow but real |

### 3.3 We Have It (competitor stronger)
| Capability | Ours | Who is stronger | Gap |
|---|---|---|---|
| **CPU throughput (Kokoro, same host)** | RTF 0.215–0.267 over HTTP | audio.cpp server RTF 0.211, warm-session | ~5–20% and shrinking; our gap is self-inflicted (§5) |
| **Per-request latency floor** | ~0.79–0.95 s for a 1-word request | streaming with 80 ms first codec frame (Qwen3-TTS), `ttft_ms` reported | **material** for interactive/animation preview |
| Memory footprint | RSS ~2.16 GB | RSS ~1.23 GB (Q8_0, warm) | 1.75× ours |
| Streaming | `StreamingEngine` iface (internal), no SSE/chunked wire format | SSE `speech.audio.delta` + chunked PCM, measurement harness | visible competitive weakness |
| Deployment story | repo-relative `libonnxruntime.so.1` + `libpiper.so` + `espeak-ng-data-v1-gpl`; must run from repo dir | self-contained binaries, GitHub Releases (Ubuntu CPU/Vulkan, Win, macOS), Homebrew, downloadable C API libs | ours is fragile |
| Model file weight | Kokoro FP32 325 MB + 56 MB side-car (381 MB) | Kokoro Q8_0 GGUF 181 MB | ours 2.1× heavier |
| Documentation of measured performance | none published | 28 report files (but GPU-centric; no x86-CPU TTS table) | neither is strong; we are silent |

### 3.4 We Lack It (adaptation candidates) — see §4 for ranking
visemes · streaming SSE/PCM with TTFT · quantized/distilled weight tier · OpenAI-compatible path names · static/pinned thread & CPU affinity (advertised `-c` but **never applied**) · per-request timing cache key · reference-audio / voice-design contract · health/version endpoint.

### 3.5 They Have It (explicitly rejected) — see §7 with rationales
143-family catalogue, 38 zero-shot cloning families, ASR/diarization/VAD/wake-word/separation/SVC/music generation, GPU backends (CUDA/Vulkan/HIP/Metal/SYCL), RKNN NPU, WASM+WebGPU browser runtime, WebUI (Studio/Arena/TTS apps), LoRA adapters, avatar lip-sync (LatentSync/SoulX), `/v1/tasks/batch`, model-manager UI.

---

## 4. Prioritized Adaptation Roadmap

Ranked by Impact ÷ Effort. Each item names its architectural fit point and at least one explicit tradeoff.

| P | Item | Impact | Effort | Architectural fit | Tradeoff / risk it introduces |
|---|---|---|---|---|---|
| **P0-1** | **Kill our own latency/throughput tax.** Apply the advertised `-c` CPU pin (currently parsed and threaded through `server.go:109`/`ui.go:221` but **never used**), set ORT `intra_op_num_threads` (no thread setting exists in the Go code), and stop spawning `piper_phonemize` **once per word** in `TokenizeWordBoundaries` (`internal/shared/shared.go:42-58` → `Phonemize` → `exec.Command` at `:456-470`). | **High** — measured red-team run: same ORT graph at 8 threads RTF 0.085 vs 0.215 through our server (≈2.5× real-world headroom); moves us ahead of audio.cpp on the metric we are losing | Days | `engine.go` contract untouched; a `RuntimeOptions` struct passed at `Initialize` + one batched `Phonemize(text)` call | Pinning to physical cores reduces multi-client headroom and needs per-model tuning tables; per-word phonemization is currently load-bearing for the VAD fallback, so batching changes word-boundary counts (regression risk in `server_timing_test.go`) |
| **P0-2** | **Harden the moat before they close it:** ship per-word **and** per-phoneme timing on the OpenAI-style route too, emit `WordsExact` (already set internally when eSpeak merges function words) in the response, and document that Kokoro/Kitten `word` entries can be phoneme groups. | **High** — converts "we're one handler better" into "our contract is richer than theirs", and prevents client-side mis-sync | Small–Medium | additive fields in `TimingResponse` (`server.go:539`) | Adding fields is a breaking-contract conversation with hyperframes/zen-board; two source-of-truths for word text (orthographic vs IPA) until alignment is fixed |
| **P0-3** | **Streaming wire format:** expose the existing `StreamingEngine` as SSE `speech.audio.delta` or chunked PCM with a first-packet target, and stream timings alongside chunks. | **High** — neutralises their most visible demo (80 ms first chunk, `ttft_ms`) for live animation/preview | Medium | `engine.go:StreamingEngine` already exists; new handler in `server.go` | Per-chunk audio breaks the "one pass, therefore always consistent" guarantee unless timings are emitted incrementally from the same duration cumsum; more state to tear down on client disconnect; doubles the playback-path test matrix |
| **P1-4** | **Distribution hygiene:** statically stage/embed `libonnxruntime` + `libpiper` + espeak data (or single-file bundle), add `/health`+version, publish CPU numbers we can stand behind. | **Medium-High** — our "just run the binary" story is currently false (repo-relative `LD_LIBRARY_PATH` loading) | Medium | `constants.go`/`config.go` path resolution | ORT is not trivially static-linkable (cgo rpath in `internal/piper/piper.go`); bundling GPL espeak-ng data keeps us on the hook for GPL redistribution even though our code is not GPL |
| **P1-5** | **Weight tier down:** ship a quantized/distilled Kokoro (or promote Kitten micro/nano as the low-CPU tier) and re-validate the duration side-car against the quantized graph. | **Medium** — 381 MB → ~190 MB, lower RSS; helps edge/Pi personas where sherpa-onnx shows Kokoro RTF 2.8–3.2 | Medium | `config.go` model registry + `getModelPaths` | Quantization can shift durations → timing drift vs the FP32 side-car; doubles the validation matrix (audio × timing × quantization) and the download/asset surface |
| **P1-6** | **Visemes (level 3)** on the same envelope, for lip-sync. Polly/Azure charge cloud-only for this; nobody local ships it (verified: 0 viseme output in audio.cpp `src|app|docs`). | **Medium-High for our persona** (hyperframes) | Medium | new field in `TimingResponse`, derived from the existing phoneme stream | Needs a phoneme→viseme table **per language**, maintenance forever; the payoff exists only if avatar/face clients actually land; word starts measured ±15–40 ms off true onset, so visemes amplify an existing precision limit |
| **P2-7** | **OpenAI-compatible aliases** (`/v1/audio/speech`, `/v1/models`, `response_format`) as thin adapters over `/tts`. | **Medium** — makes us a drop-in inside stacks already wired for audio.cpp/Kokoro-FastAPI; a distribution wedge | Small | wrapper routes, no engine change | Two public API shapes to keep stable + docs/CORS/auth surface; risks clients depending on the OpenAI shape and never discovering our timing fields (put them in a non-standard key and document loudly) |
| **P2-8** | **Decide voice cloning: extend `Synthesize(text, voice, speed)` with a reference/embedding artifact, or declare it out of scope in writing.** | **High for strategy, Low for code** | Architectural | breaks the current engine contract; every engine must answer it | True zero-shot cloning is 38 families of their moat we cannot match — a half-hearted Kokoro style-vector blend invites "fake cloning" criticism; a documented "not our market" line is free but caps TAM |

---

## 5. Measured Evidence (this session, same host: Ryzen 7 9700X 8C/16T, 60 GB, no GPU, idle during timing) [VERIFIED]

| Engine / path | Model + precision | Audio | Median wall | RTF | RSS |
|---|---|---|---|---|---|
| audio.cpp **server**, `--threads 8` | Kokoro-82M **Q8_0 GGUF**, 181 MB | 6.53 s | 1.38 s | **0.211** | 1.23 GB |
| audio.cpp CLI, threads 4 / 8 / 16 | same | 6.53 s | 2.46 / 1.81 / 4.82 s | 0.377 / 0.277 / **0.739** | 1.21 GB |
| zen-tts HTTP, ORT default (16 logical) | Kokoro v1.0 **FP32 ONNX** 325 MB + 56 MB side-car | 7.05 s | 1.88 s | 0.267 | 2.16 GB |
| zen-tts HTTP, re-run after idle | same | 7.55 s | 1.62 s | 0.215 | — |
| zen-tts raw ORT graph, 8 threads (no HTTP) | same graph | 7.15 s | ~0.6 s | **0.085** | — |
| zen-tts HTTP, 1-word request "Hi." | same | ~1.3 s | **0.79–0.95 s** | ~0.7 floor-bound | — |
| zen-tts HTTP, `?timestamps=2` vs plain | same | 7.05 s | 2.10–2.20 vs 2.10–2.37 s | **timing envelope ≈ free** | — |
| Build | | | audio.cpp **196 s** (-j16, 52 MB CLI + 60 MB server, 482 MB tree) | | zen-tts **4.9 s**, 25 MB binary |
| Runtime deps | | | audio.cpp: `libgomp/libstdc++/libc/libm` only | | ours: `libonnxruntime.so.1`, `libpiper.so`, `libasound` from repo-relative paths |

Re-run: `/tmp/audio.cpp/build/bin/audiocpp_cli --task tts --family kokoro_tts --model /tmp/models-gguf/kokoro-82m-q8_0.gguf --backend cpu --threads 8 --language en-us --voice-id af_bella --metrics --words-out /tmp/words.json --text "<sentence>"` vs `go build -o /tmp/zen-tts-bench . && /tmp/zen-tts-bench -m kokoro-v1.0 -p <port>` + `curl /tts`.

**Timing feature probe, side by side.** audio.cpp `--words-out`: 18 entries `{start_sample, end_sample, word, confidence}`, `word` values are IPA groups (`"ðə"`, `"kwˈɪk"`, `"bɹˈaʊn"`), `confidence` hardcoded 0; zero occurrences of `phoneme` in `app/cli`/`app/server`. Ours: 19 `timings` + 94 `phonemes` + `timing_source="duration_graph"` + `frame_rate=30` + `frame_ms=25`. Caveat: our first `timings` entries are *also* IPA groups (`"ðə"`, `"kwˈɪk"`) — same eSpeak merging (`shared.go` alignment grouping on the space token), so D2 beats them but D-accuracy is a shared constraint, not a win.
Caveat: the checked-in local `./zen-tts` binary is from Jun 9 and does **not** serve `?timestamps` — all timing numbers above came from a fresh `go build` (untracked artifact, not a repo issue).

---

## 6. Honest Weaknesses in Our Own Story

### 6.1 The moat is one handler deep *for Kokoro only*
Their Kokoro session already produces the duration tensor (`src/models/kokoro_tts/predictor.cpp:765-820` → `session.cpp:483-580 append_kokoro_word_timings()`, opt-in `return_timestamps`), and `/v1/tasks/run` already returns base64 audio **plus** `words` in one body (`app/server/server_protocol.cpp:1042-1206`). Only `/v1/audio/speech` drops it (`app/server/runtime.cpp:919-944`). **The durable part of the moat is the other three engines:** their `model_specs/kitten_tts2.json`, `piper_tts.json`, `inflect_v2.json` declare no duration/timestamp capability, so matching us there needs new engine work, not plumbing — plus they have no frame mapping or provenance field at all.

### 6.2 Only 2 of 4 engines deliver the headline claim
Kitten `duration_graph` ✔ · Kokoro `duration_graph` (with the hand-patched side-car) ✔ · **Piper `vad_estimate`** ✖ exact · **Inflect `none`** ✖. Marketing "frame-locked" must be scoped to the engines that qualify, and VAD estimates are silence analysis (±10 ms resolution, no true onset guarantee).

### 6.3 Total cost of ownership is *not* low
96 MB vendored `piper/` (two ONNX Runtime versions, `libpiper`, GPL `espeak-ng-data-v1-gpl`), a manually re-exported + TopK-patched Kokoro duration side-car, a Japanese kagome dictionary, third-party weight auto-download at first run and voice fetch mid-request. audio.cpp has no ONNX Runtime in its inference path at all. This is the strongest sunset argument (§8) and we should reduce it, not deny it.

---

## 7. Explicit Rejections (they have it, we will not chase it)

| Competitor feature | Rejection rationale (one line) |
|---|---|
| 143 model families / 250 variants | Chasing catalogue breadth turns us into a slower copy of a 4.1k-★, weekly-release project with a C++ engine-per-model cost curve we cannot staff |
| 38 zero-shot cloning families + base64 `voice_ref` | Wrong buyer: animation/caption sync needs deterministic timing, not timbre; also drags in consent/moderation surface — but see P2-8, we must *state* the boundary |
| ASR, diarization, VAD, wake-word, source separation, SVC, music generation | Scope creep: each is a separate product with its own accuracy expectations; our HTTP surface stays one endpoint |
| GPU backends (CUDA/Vulkan/HIP/Metal/SYCL) | Our persona is CPU-constrained app hosts; GPU is their strength and would double our CI matrix for no timing-accuracy gain |
| RKNN NPU, WASM+WebGPU browser builds | Separate toolchains and release trains; HeadTTS already owns the browser niche |
| Embedded WebUI (Studio/Arena/voice library) | The TUI *is* our control plane; a web front end duplicates hyperframes/zen-board's own UI |
| LoRA adapters on TTS models | High maintenance, model-specific, and irrelevant to frame-locked alignment |
| Avatar lip-sync video generation (LatentSync/SoulX) | Consumes visemes — which is our product's *output*; sell the stream, don't build the face |
| `/v1/tasks/batch` long-form batch jobs | Long-form narration quality is a different engine concern; risk of becoming an audiobook pipeline |

---

## 8. Tradeoffs & Risks of the Recommended Path

1. **Timing-first positioning vs. model breadth expectations.** Marketing D1–D3 invites "but it only has 4 engines" objections from local-AI users. Mitigation: publish the capability matrix openly and point at audio.cpp for breadth workloads — a deliberate "we are the sync engine, not the audio omnibus" line.
2. **P0-1 pinning vs. multi-client serving.** Capping ORT threads raises single-request speed on our 8-core host but lowers throughput under concurrent clients; we need a documented default (`auto` = physical cores) instead of a hardcoded number, and the per-word phonemization change touches the VAD fallback used by Piper, so `server_timing_test.go` regressions are the real cost.
3. **Streaming (P0-3) undermines our own strongest claim.** The "audio and timings always come from one pass" guarantee becomes "chunks arrive incrementally with partial timings" — clients may treat streamed timings as final. Requires `timing_source` semantics per chunk, and a `partial` flag (contract churn).
4. **Quantization (P1-5) risks the alignment we sell.** Durations measured on FP32 may not hold on a quantized graph; if the timing drifts from the waveform we lose the one property nobody else has. Gate behind an explicit "timing-validated" asset list.
5. **Visemes (P1-6) inherit an accuracy ceiling.** Word onsets are already ±15–40 ms from true acoustic onset; at 30 fps that is ~1 frame, at 60 fps ~2. A viseme stream implies lip-sync quality we have not demonstrated. Publish the error budget with it or don't ship it.
6. **GPL exposure grows with distribution convenience.** Bundling `espeak-ng-data-v1-gpl` (GPL-3.0) and per-model licenses (KittenTTS 2 Stellon Community License, Piper voices non-commercial) is a real constraint on any future "download & ship voices" story.
7. **The single biggest strategic risk:** if audio.cpp spends one weekend wiring `word_timestamps` + a `phonemes` array into `/v1/audio/speech`, our differentiator becomes "contract quality + 3 engines + Go packaging". Our hedge is to make the *contract* the thing worth copying (frame mapping, provenance, per-engine matrix, `WordsExact` honesty) and to measure/beat their throughput on the metric they will show off, which P0-1 does cheaply.

---

## 9. Decision

**Continue zen-tts**, reframed as **the local frame-locked speech-timing engine** (one binary, four engines, animation-grade alignment contract), not as "a local TTS server that competes with audio.cpp on models or speed".

Conditions to accept (kill-criteria if they fail):
1. P0-1 lands within the next cycle and puts us at ≤ audio.cpp RTF on Kokoro CPU (currently 0.211 measured here).
2. P0-2 lands so that our envelope is provably richer than anything they can plumb through (`WordsExact`, phoneme arrays, `timing_source`, frame mapping).
3. Cloning is explicitly declared in or out of scope (P2-8) so the "4 engines vs 143" comparison stops applying to us.

**Steelman for the alternative (sunset + upstream):** their Kokoro timing already exists end-to-end except for one handler; one PR would buy us frame-locked timing, streaming, cloning and 143 families of maintenance we would not pay, inside a 4.1k-★ weekly-release project — and our Go/TUI/userscript packaging is not a moat, it is a preference. **Why we still continue:** that PR gives them *Kokoro* timings, not our contract; their Piper/Kitten/Inflect specs declare no duration capability at all; there is no frame/`timing_source`/provenance concept anywhere in their payload; and our cost of iteration is 4.9 s builds against their 196 s + 482 MB tree. We keep the product, stop pretending we are in the model-count or throughput race, and convert the timing envelope into the interface other tools standardise on.
