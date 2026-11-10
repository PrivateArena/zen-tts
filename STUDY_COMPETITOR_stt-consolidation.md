# Competitive Study 2 — STT consolidation: do we keep zen-stt and zen-tts, or standardise on `audio.cpp`?

Companion to `STUDY_COMPETITOR_audio-cpp.md` (TTS side, same session). This study covers the speech-to-text side and the two-server complexity question.
Evidence sources: local build of `/tmp/audio.cpp` @ `0382d37` (v0.9.4), `zen-stt/PROJECT_OVERVIEW.md`, and an expert web query to Claude via `zchat` (transcript at `/tmp/claude_ans.md`, full 9-part answer; its "I don't know" items are below re-resolved from the local repo where we could).

---

## 1. Executive Summary & Decision

**Short answer: split the decision.**

| | Verdict | One-line reason |
|---|---|---|
| **zen-stt** | **RETIRE the inference core.** Keep only a thin dataset-orchestration tool (ffmpeg clip extraction + LJSpeech `metadata.csv` + Vietnamese text normalisation) that drives `audiocpp_server`/`audiocpp_cli`. | We are structurally slower (hand-written greedy decoder with **no KV cache**, O(N²) re-feed), we get **no word timing at all** (we force `<|notimestamps|>`), and the ASR problem is better solved by a maintained runtime we already vendor behind a build tag. |
| **zen-tts** | **KEEP, in reduced scope** (per Study 1): the frame-locked word+phoneme timing envelope server. Do **not** chase model breadth or STT. | audio.cpp has **no per-phoneme timing and no timing on the OpenAI-style speech route**; Claude independently reached the same conclusion from the public docs ("Not yet: zen-tts"). Our timing contract is the only asset we cannot buy back with an upstream PR. |
| **Two-server complexity** | **Goes down, not up**, with this split: one third-party binary doing everything heavy (TTS breadth + ASR + VAD + diarization) + one small Go server we own for the animation timing contract. | Today we own **two** bespoke inference stacks (Whisper-ONNX decoder loop and Kokoro/Kitten/Piper/Inflect orchestration). Retiring the weaker of the two removes the larger maintenance liability. |

**Answers to the two things you asked directly:**
- *"They can run STT, and efficiently?"* — **Yes.** Verified CPU (x86) figures in their own docs: SenseVoice `RTF 0.033–0.036` (27.9–30.2× real time, peak RSS 279 MiB) `docs/archive/sense_asr_pr.md:177-190`; Parakeet-TDT `10.4× real time` on CPU 12 threads `docs/models/parakeet_tdt.md:301-306`; Kroko `RTF 0.045–0.053` (19–22×), RSS 1345 MiB `docs/reports/kroko_asr_validation.md:143-174`. Their server returns `rtf` per request, so this is measurable, not marketing. `[VERIFIED in local repo]`
- *"We're straight up slower?"* — **Yes on STT, and it is structural.** Our decoder re-encodes the whole token sequence every step (`decoder.go` "Full re-feed decoder, no KV cache"), on 30 s padded windows, per VAD segment, with a Go-side BPE decode and a gonum Mel STFT. Their ASR paths are KV-cached/CTD-TDT implementations benchmarked in-process. Our Whisper-Large-v3-Turbo ONNX at FP32 on this class of CPU cannot reach the ~20–30× their small models report, and we would have to re-implement KV caching, beam search and Mel in ONNX to approach it.
- *"They can also expose timing, right?"* — **On ASR yes, but not for Vietnamese, and it does not help our TTS moat.** Details in §4 and §6.3 — this is the one place where the honest answer changes the decision.

---

## 2. Our Specialization (STT side) and its real cost

**What zen-stt is:** a Go + ONNX-Runtime Vietnamese *dataset builder* — ffmpeg pipe → Silero VAD v5 → 30 s windows → 128-band log-Mel → Whisper Large v3 Turbo encoder/decoder → BPE → LJSpeech `(clip, transcript)` pairs; plus a real-time `/stt` HTTP endpoint. Dual backend: pure-Go ORT (default) or CGO sherpa-onnx behind `-tags sherpa`.

**Genuine value (worth keeping):**
- **The LJSpeech dataset contract** (`internal/dataset/writer.go`) — id/clip/`metadata.csv` schema tuned to our Vietnamese TTS voice-training loop. This is domain knowledge, not code.
- **Vietnamese text handling** in the transcription→dataset path (forced `<|vi|>` language, normalisation/QA filters).
- **The `build` batch pipeline shape**: one command, audiobook in, training set out.
- ffmpeg-any-format decode into float32 with no temp files (audio.cpp's core routes are **WAV-only**; MP3/FLAC need an optional `audio_decode` frontend module — `app/server/README`, verified locally; Claude reached the same conclusion).

**Architectural liabilities (the reason to retire it):** `[VERIFIED from PROJECT_OVERVIEW.md]`
1. **No KV cache** — full-sequence re-feed per decode step, O(N²) over ≤448 tokens, self-described as "acceptable on Ryzen AI MAX 395+" (i.e. tuned for one strong machine).
2. **`<|notimestamps|>` is hardcoded** — we throw away the token-level timestamps Whisper natively emits. So our STT contributes *zero* timing value while our TTS contributes all of it.
3. **Hand-rolled DSP/tokenisation** (Slaney Mel filterbank, gonum FFT, BPE over `vocab.json`) — each is a silent-corruption surface if it drifts from the reference export, and it is the main reason the project is "not well tested".
4. **Two inference backends** (pure-Go ORT + CGO sherpa-onnx behind build tags) → two build matrices, two vendored runtime trees, and a `piper/` symlink convention inherited from zen-tts.
5. We are **already running sherpa-onnx** as an option — i.e. the project has implicitly admitted the runtime should not be ours.

---

## 3. STT Competitor Landscape

| Product | Stack | Traction | STT timing output | Vietnamese | CPU story |
|---|---|---|---|---|---|
| **audio.cpp** (consolidation target) | C++/ggml, GGUF, `asr`+`align`+`vad`+`diar` tasks, OpenAI-ish `/v1/audio/transcriptions{,/details,/live}`, `/v1/batches/transcriptions`, `/v1/audio/alignments`, SSE live PCM | 4,150 ★, weekly releases, Apache-2.0, Unsloth Studio | `words[] {word,start_sample,end_sample,confidence}` + `segments[]` + `speaker_turns[]` + `timing{wall_ms,audio_duration_ms,rtf}` | `vi` declared in `qwen3_asr`, `cohere_asr`, `nemotron_asr` (`vi-VN`), `confucius4_r2t2`, `vibeasr`; `sense_asr` maps `vi`→`auto` only | First-class `--backend cpu`; x86 ASR RTFs as above; **no ARM/aarch64 prebuilt** |
| **sherpa-onnx** (already our optional backend) | C++/ONNX Runtime, 48+ TTS langs, huge ASR set incl. Whisper/Parakeet/ZipVoice | 15,862 ★, Apache-2.0 | ASR token timestamps only | Yes (Whisper/multilingual models) | CPU/embedded-first — but no HTTP server; library + SDKs |
| **whisper.cpp** | C/C++ Whisper, quantized GGUF | 55,067 ★, MIT | Word-level timestamps (`-ml 1`, "experimental"), segment timestamps | Yes (any Whisper language incl. `vi`) | CPU-adequate, NPU/CoreML/Vulkan/CUDA offloads — verified `README.md:347,164,703-705` |
| **faster-whisper** | CTranslate2, Python | 26,647 ★, MIT | Word timestamps, VAD built in | Yes | ~4× openai/whisper claim (their repo) — the incumbent a hobby pipeline like ours is usually replaced by |
| *(reference)* audio.cpp forced aligners | `qwen3_forced_aligner`, `mms_forced_aligner`, `crisperwhisper --task align` | — | word/char spans from a known transcript | **Qwen3 aligner: no `vi`** (11 langs: zh,yue,en,de,es,fr,it,pt,ru,ko,ja). MMS: 1130 langs incl. `vie` but **CC-BY-NC-4.0** + caller must supply romanization | MMS CPU: 0.13× RT (short) / 0.33× (long-form), 2.1 GB |

---

## 4. Feature Comparison Matrix (STT)

Legend: **Ours** = zen-stt today · ✔ parity · **STRONG** = our differentiator · ~ present but weaker · ✖ missing · **N/A** deliberately out of scope.

| Capability | Ours (zen-stt) | audio.cpp | sherpa-onnx | whisper.cpp | Class |
|---|---|---|---|---|---|
| Vietnamese transcription | ✔ Whisper-Large-v3-Turbo, forced `vi` | ✔ 5 families declare `vi` (quality **unverified** — no vi WER published by them) | ✔ | ✔ | Parity |
| **Word timestamps from ASR** | ✖ **disabled on purpose** (`notimestamps`) | ✔ `--words-out` / `/details` (model-internal for parakeet/kroko/nemotron/sense) | ✔ (tokens) | ~ experimental | **We lack it** → they are stronger |
| Per-word timestamps **for Vietnamese** | ✖ | ✖/uncertain: aligner has no `vi`; `cohere_asr` explicitly "timestamps … not supported"; MMS is NC-licensed | ✔ via Whisper token times | ✔ via segment/word times | Niche gap nobody cleanly fills |
| Long-form / VAD-integrated segmentation | ✔ our own Silero+30 s windows | ✔ internal Silero in `qwen3_asr`, `audio_chunk_mode=auto`, `--vad-chunks-out` (≤45 s plans), `parakeet_tdt offline_mode=long_form` | ~ | ~ | Parity (theirs is maintained) |
| Speaker diarization | ✖ | ✔ `nemotron_3_diar` (≤8 spk), `sortformer_diar(_v2)`, `moss_transcribe_diarize`, SegLST with `speaker/start_time/words` | ~ | ✖ | **Adaptation candidate → rejected** (§6) |
| Streaming / live ASR endpoint | ~ `/stt` upload-only HTTP | ✔ `/v1/audio/transcriptions/live`, SSE, `vibevoice_asr_streaming`, `--audio -` raw PCM stdin | ✖ (library) | ~ real-time example binary | They are stronger |
| Batch API | ~ `build` subcommand (files) | ✔ `/v1/batches/transcriptions`, `/v1/tasks/batch`, `--batch-audio-dir` | ✖ | ✖ | Parity-ish |
| **(clip, transcript) LJSpeech export** | **STRONG** — the whole point | ✖ **NOT FOUND** (`--clips-out` absent; only `--words-out`/`--segments-out`; `convert_audio` pipeline step crops but shells to ffmpeg) | ✖ | ✖ | **Our Core Advantage** |
| Any-format input (MP3/FLAC) in one binary | ✔ ffmpeg pipe | ~ WAV-only core; MP3/FLAC need optional frontend module | ✔ (own decoder) | ~ | We have it (comparable/better) |
| Single-binary deploy + TUI | ✔ Go, but needs `libonnxruntime.so` + `piper/` tree | ✔ 52–60 MB binaries, no `.so` deps, WebUI, Docker CPU/CUDA | ✔ libs | ✔ | Parity |
| KV-cached / non-quadratic decoding | ✖ **structural liability** | ✔ per-family implementations | ✔ | ✔ | **We lack it (won't fix — retiring)** |
| Published CPU ASR RTF for our stack | ✖ | ✔ (SenseVoice/Parakeet/Kroko numbers) | ✔ (Pi RTF tables) | ~ | They are stronger |
| ARM/RK3588/Pi ASR evidence | ✖ | ✖ **NOT FOUND** (TTS-only NPU docs, no aarch64 prebuilt) | ✔ the only one with Pi numbers | ~ | Only sherpa-onnx covers edge |

---

## 5. What the Claude (zchat) Query Contributed — and where we overrode it

Claude read the public README/`app/server/README.md`/`api.md`/`model_licenses.md` and answered conservatively. Value and corrections:

| Claude's position | Status after local verification |
|---|---|
| "No plain Whisper family in audio.cpp; `qwen3_asr` is the only core-tree `vi` family with a full GGUF ladder + streaming" | **Confirmed** against `model_specs/qwen3_asr.json:28`, `README.md` ASR table. Best Vietnamese path. |
| "`/v1/audio/transcriptions/details` returns `words[]{word,start_sample,end_sample,confidence}`, `segments[]`, `speaker_turns[]`, `language`, `sample_rate`, `timing{wall_ms,audio_duration_ms,rtf}`" | **Confirmed** (`app/server/api.md:229-246`, `server_protocol.cpp`). Offsets are **samples**, not seconds — same convention gap our TTS API already solved with `frame_ms`/`frame_rate`. |
| "`qwen3_forced_aligner` covers 11 languages, **no `vi`**" | **Confirmed** from `model_specs/qwen3_forced_aligner.json` ("11 supported languages", list matches). Decisive for our Vietnamese dataset pipeline. |
| "`--words-out`: I don't know"; "published CPU RTF for ASR: I found none"; "CUDA is the primary optimization target" | **Overridden by local repo**: `--words-out` exists (`app/cli/main.cpp:170,950`) and we executed it; per-model CPU ASR RTFs are documented in `docs/models/*` and `docs/reports/kroko_asr_validation.md` (their **top-level README** is GPU-centric, which is why the doc-only read saw nothing). The "CUDA primary" quote is real (`app/server/README.md`) and is our strongest counter-argument for keeping *anything* CPU-critical in-house. |
| "`/v1/audio/speech` timing: treat `duration_graph` as unavailable until you check the code" | **Confirmed by us locally**: duration timings exist per-engine and reach `/v1/tasks/run` JSON, but **not** the OpenAI-style speech route, and **no per-phoneme array anywhere** in `app/cli` or `app/server`. Study 1's conclusion holds. |
| Recommendation: "Deleting both Go servers is not reasonable. Replace zen-stt's inference; keep zen-tts; keep chunking, LJSpeech writing, VN normalisation, and a version-pinned build in-house." | **Adopted**, with two additions: (1) their `/v1/audio/transcriptions` accepts WAV only, so our ffmpeg decode step is not optional — it is the adapter; (2) insist on a version pin + gold-set WER gate because pre-1.0 API churn is real (`vietneu_tts`→`vieneu_v3_turbo` rename, deprecated model manager, Kokoro `speed` needing a post-2026-09-22 GGUF). |

---

## 6. Prioritized Consolidation Roadmap

| P | Action | Impact | Effort | Architectural fit | Tradeoff / risk |
|---|---|---|---|---|---|
| **P0-A** | **Gate before commitment:** run `audiocpp_cli --task asr --family qwen3_asr --request-option language=vi` (and `cohere_asr`) on 2–3 chapters of the real audiobook + our gold transcripts; record WER/CER, wall time, `rtf`, RSS. Pin the audio.cpp version. | **High** — decides the whole question with ~1 hour of work instead of an opinion | Small | None required; CLI-only test | One-off cost of building their binary (196 s here) + downloading a 0.6B GGUF; if it loses, we keep a codebase we want to delete anyway |
| **P0-B** | **Retire `internal/whisper` + `internal/vad` + `internal/audio/mel.go` + tokenizer**; replace with one STT client (POST `/v1/audio/transcriptions`, or shell `audiocpp_cli`) behind the existing `whisper.Backend` interface (`backend.go` is already the seam — the swap is confined to a new `backend_audio.cpp` implementation) | **High** — deletes the O(N²) decoder, the hand-rolled Mel/BPE, and the CGO sherpa build tag | Medium | Fits cleanly: `Backend{Transcribe,Close}` already decouples engine from pipeline | We trade a Go-only toolchain for a C++ dependency + separate process/HTTP port + their model-manager artefacts; loss of "one static binary" for dataset builds; their CPU-only default is 1 request per model at a time |
| **P0-C** | **Keep the dataset sink in-house**: LJSpeech clip extraction (ffmpeg `-ss/-t` from VAD/segment spans) + `metadata.csv` + Vietnamese normalisation/QA filters | **High** — this is the only STT capability audio.cpp does *not* have (§4) and it is our actual product requirement | Small | Stays as-is; input becomes their `segments[]` JSON | We now own a JSON-schema dependency on their `start_sample/end_sample` semantics (samples, not seconds; `confidence` may be a constant) — version-pin and assert on parse |
| **P1-D** | **Fold the real-time `/stt` endpoint into the audio.cpp server route** (`/v1/audio/transcriptions/live` + `--audio -` PCM), and delete `server.go` in zen-stt | **Medium-High** — removes one of our two Go servers outright | Small-Medium | None: we become a client | Loses our request-level control (queueing, per-request language forcing) and adds an ops dependency for real-time UX; their live route is SSE, our clients expect our JSON shape → thin shim needed |
| **P1-E** | **Re-scope zen-tts to the timing contract only** (Study 1 P0-1 thread/pin fix, P0-2 `WordsExact`+provenance hardening, P0-3 streaming) | **High** | Days–Medium | `engine.go` interface layer unchanged | Explicitly accepting "4 engines, no breadth" positioning invites the wrong comparison; needs the §7 rejection list published |
| **P2-F** | **Move the STT client under a single `zen-audio` tool** (or into zen-board), so the repo count goes 3 → 1 Go audio repo + audio.cpp | **Medium** — real maintenance-burden reduction, which is the stated goal | Medium | New repo name after domain: dataset building + STT client | One more rename/migration cycle; git-history and Makefile/model-symlink conventions from both repos must be reconciled |
| **P2-G** | Re-evaluate **Vietnamese word-level timestamps**: only if captions/forced-alignment become a product need; today clip-level text is sufficient for LJSpeech | Low now / High later | Medium | Would be a new client-side step calling `--task align` | MMS aligner is **CC-BY-NC-4.0** (blocks commercial use); Qwen3 aligner has no `vi`; Whisper word times are available in sherpa-onnx — a third option we'd rather not fork for |

**Explicitly: do NOT re-implement KV caching, beam search, or Whisper token timestamps in Go.** That is the trap this study exists to close.

---

## 7. Explicit Rejections (STT side)

| audio.cpp/whisper.cpp feature | Rejection rationale |
|---|---|
| Owning any ASR inference code (decoder loop, Mel, BPE) | Not our product; 55k-star and 4.1k-star implementations exist and ship weekly — our version is the slow, untested one |
| 31 ASR families / diarization / emotion / wake-word / music | Dataset building needs one Vietnamese-accurate model; more families = more licence and validation surface |
| Speaker diarization (`nemotron_3_diar`, `sortformer_diar`) | Audiobooks are single-narrator per file; adding speaker labels is a feature we have no consumer for (revisit if multi-cast drama datasets are needed) |
| `/v1/audio/alignments` forced-alignment in the dataset path | No Vietnamese-capable, commercially-usable aligner exists in their catalogue — building on it would encode a licence violation or a romanization hack |
| Their WebUI / model-manager UI | Our workflow is a CLI batch command; a browser tab is not an interface for an unattended dataset build |
| Whisper word-level timestamps via audio.cpp | We need **clip-level** `(clip, text)` pairs for TTS training; per-word times add a mapping problem we would have to solve again in the writer |
| ARM/RK3588 ASR with audio.cpp | No aarch64 prebuilt and no ARM ASR numbers — if edge ASR ever matters, sherpa-onnx is the verified path, not this one |

---

## 8. Tradeoffs & Risks of the Recommended Path

1. **Unmeasured Vietnamese quality is the whole ballgame.** audio.cpp publishes no `vi` WER; our Whisper-Large-v3-Turbo is known-good for our corpus. Retiring zen-stt before P0-A means possibly regressing our TTS voice-training data silently. *Mitigation: gold-set gate, keep the Go decoder on a frozen branch (no further investment) until the gate passes.*
2. **We trade Go-only for a C++ + model-artefact dependency chain** (build or download binaries, GGUF weights, optional frontend modules for MP3/FLAC, `libgomp`). Total maintenance goes down, but *deployment* complexity goes up and one more process is running on the box — partially self-defeating the "two servers is too complex" motive unless P1-D actually deletes our HTTP layer.
3. **Pre-1.0 API churn.** Their payloads and family names move (`vietneu_tts`→`vieneu_v3_turbo`, deprecated model manager, date-gated Kokoro GGUF behaviour). Our parser must pin a version and fail loudly on schema drift, not degrade silently.
4. **Non-commercial licences in the tempting options.** `crisperwhisper` (non-commercial, restricts *outputs*), `mms_forced_aligner`/`hviske_asr`/`sortformer_diar` (CC-BY-NC-4.0). Anything we pick for a commercial dataset build must come from the Apache-2.0 / MIT / CC-BY / OpenMDW rows: `qwen3_asr`, `cohere_asr`, `nemotron_asr`, `parakeet_tdt`, `silero_vad`.
5. **"CUDA is the primary optimization target; performance and model coverage differ on other backends"** (their own server README, quoted by Claude and consistent with the absence of any x86-CPU TTS benchmark). Our TTS CPU numbers were parity-to-slightly-behind locally; do not assume their CPU path improves for free as they add families.
6. **Concurrency ceiling:** default one request per model at a time. Fine for batch dataset building; a ceiling for any shared/team STT service.
7. **Strategic risk of the split:** after this, our *only* in-house audio asset is the TTS timing envelope. If that contract is not productised (frame mapping, provenance, visemes, streaming) it will look like a 25 MB vestigial server in two years and be deleted by the next efficiency pass. That is the argument for Study 1's P0 items, not for keeping STT alive.

---

## 9. Verification Plan (cheap, before any deletion)

```
# 1. build once (measured here: 196 s, -j16, CPU-only, no deps beyond GCC13+cmake)
cmake -S /tmp/audio.cpp -B /tmp/audio.cpp/build && cmake --build /tmp/audio.cpp/build -j16 --target audiocpp_cli audiocpp_server
# 2. Vietnamese A/B on the same chapter
audiocpp_cli --task asr --family qwen3_asr --model <qwen3_asr_0_6b_q8_0.gguf> --backend cpu --threads 8 \
  --audio chapter.wav --language vi --text-out ac.txt --words-out ac.json --metrics
# vs
zenstt build --in chapter.wav ...   # current gold path
# 3. compare WER/CER against our gold transcript, wall time, --metrics rtf, peak RSS
```
Gate: adopt STT retirement iff `WER(audio.cpp vi) ≤ WER(our Whisper) + 1.5 pt` **and** `RTF ≤ our current RTF`. Otherwise: freeze zen-stt (no features, security/dep updates only) and re-test at their next release cadence (≈weekly).
