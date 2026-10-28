# VERIFICATION: Kokoro / KittenTTS speech-timing claims

Target claim: `CONFIRM_REQUIRED_tts-speech-timing.md`
Date: 2026-10-28 · Host: 16-core CPU, no GPU · Everything below was executed locally, artifacts in `scratch/timing/`.

Environment installed for the test: `scratch/pytts` (uv venv, `torch 2.14.1+cpu`,
**kokoro 0.9.4** with `hexgrad/Kokoro-82M` = **v1.0 weights**, **kittentts 0.1.3**, `onnxruntime`,
`faster-whisper`), production server `./zen-tts -p 5055 -m kokoro-v1.0` (pid 2101735, port 5055).

---

## 1. Verdict on each claim

| # | Claim in `CONFIRM_REQUIRED_tts-speech-timing.md` | Verdict | Measured evidence |
|---|---|---|---|
| C1 | Kokoro Python (`KPipeline`) returns `result.tokens` with `start_ts` / `end_ts` per word, "without requiring an external post-inference alignment step" | **TRUE** (kokoro ≥ 0.9.x) | `scratch/timing/py_kokoro_timings.json`: every word token carries both fields. Source: `kokoro/pipeline.py:314-319` (`KPipeline.join_timestamps`) writes them from `KModel.Output.pred_dur`; the fields live on `misaki.token.MToken`. |
| C2 | "explicit time boundary metadata" is exact | **PARTLY TRUE** | Values come from the duration predictor and are quantized to **half mel-frames = 12.5 ms** (1 frame = 25 ms = 600 samples @ 24 kHz). Word *spans* absorb the following space/punctuation
frames (Python tokens: `dog` ends 3.025 s while its audio is silent from 2.83 s; duration graph: `dog,`
ends 3.300 s while audio is silent from 2.93 s). Use word **starts** (±15-40 ms measured) or `start[i+1]`
as the previous word's end. |
| C3 | Kokoro is non-autoregressive (StyleTTS2) with a duration predictor computed before synthesis | **TRUE** | `kokoro/model.py:104-110`: `pred_dur = round(sigmoid(duration_proj).sum(-1)/speed).clamp(min=1)`, and `sum(pred_dur) * 600 == audio samples` exactly (verified: 201 frames ↔ 120,600 samples). |
| C4 | ONNX/web Kokoro runtimes do **not** expose durations | **TRUE** | `models/kokoro/kokoro-v1.0.onnx` graph outputs: `['audio']` only. |
| C5 | "KittenTTS: Yes — duration predictor output, phoneme/frame granularity" | **TRUE, and stronger than claimed** | Our shipped `models/kitten/kitten-tts-mini.onnx` outputs **`['waveform', 'duration']`**; `duration` is `int64[seq_len]`, **600 samples/frame**, and `sum(duration)*600 == len(waveform)` **exactly** (e.g. 229 frames ↔ 137,400 samples). Same in `KittenML/kitten-tts-nano-0.1`. `internal/kitten/engine.go:127-136` requests **only `waveform`** → the alignment is thrown away. |
| C6 | "Piper: Yes — phoneme durations via ONNX/C++" | **FALSE for the ONNX exports we ship** | `models/piper/*.onnx` outputs: `['output']` (audio only). VITS durations exist only at training time (MAS), the inference graph does not return them. |
| C7 | "Determinism: fixed duration outputs per phoneme for a given random seed, so audio buffer and timing metadata are deterministic and locked together" | **PARTLY FALSE — needs a mitigation** | `pred_dur` is deterministic (verified bit-identical across runs). The **audio is not**, in two of three paths:<br>· Kokoro **torch** (Python): two runs of the same input differ, max abs sample error **0.047** (≈ −26 dB) — `kokoro/istftnet.py:205,253` inject `torch.randn_like` noise. Needs `torch.manual_seed`.<br>· KittenTTS **ONNX**: waveform **not** bit-identical across runs (graph contains `RandomNormalLike` + `RandomUniformLike`) while `duration` **is** identical.<br>· Kokoro **ONNX** (the one zen-tts uses): bit-identical across runs. |
| C8 | frame mapping `frame_index = floor(t_seconds × FPS)` | TRUE (trivial) | With 25 ms frames: at 30 fps a token edge is ±0.75 frame; at 60 fps ±0.4 frame. Word starts measured to ±15-40 ms of true acoustic onsets → good for caption highlighting, not for phoneme-exact viseme sync. |
| C9 | Bark/XTTS need a forced aligner | NOT TESTED | No such model installed; out of scope. |

**Bottom line:** the *capability* the claim describes is real and it is even closer at hand than the
claim says — the KittenTTS model already in this repo emits exact per-phoneme durations, and a Kokoro
durations ONNX can be exported that agrees with the production audio **frame-for-frame**. What the
claim gets wrong is determinism and Piper, and it never mentions that zen-tts' own current timing source
is a lossy VAD heuristic that is ~0.5 s off.

---

## 2. What zen-tts on :5055 actually does today

| Test | Result |
|---|---|
| `POST /tts` (kokoro, af_bella) | 200, 24 kHz mono 16-bit, 5.050 s for the reference sentence. |
| `POST /tts?timestamps=1` (kokoro) | 200, JSON `{audio(b64), timings, duration}`; **audio synthesized twice** (`server.go:388-402` then `:447`). |
| `POST /tts?timestamps=1` (kitten) | `timings: null` — `KittenEngine` implements no `TimingEngine` (`internal/kitten/engine.go` has no `SynthesizeWithTimings`). |
| Timing quality | **Unusable for frame-accurate work.** See table below. |

Reference sentence `"The quick brown fox jumps over the lazy dog, then runs away fast."`,
silences measured directly from the server's own PCM (`silence_anchor_table.py`):

| word | **duration graph (proposed)** | `?timestamps=1` (current) | Whisper ref |
|---|---|---|---|
| The | 0.425 → 0.475 | **0.000 → 0.200** | 0.000 → 0.440 |
| quick | 0.525 → 0.725 | 0.200 → 0.588 | 0.440 → 0.560 |
| brown | 0.775 → 1.125 | 0.588 → 0.977 | 0.560 → 0.880 |
| fox | 1.200 → 1.575 | 0.977 → 1.365 | 0.880 → 1.220 |
| jumps | 1.650 → 1.950 | 1.365 → 1.580 | 1.220 → 1.680 |
| over | 1.975 → 2.100 | **1.580 → 3.000** | 1.680 → 2.040 |
| the | 2.125 → 2.175 | **3.000 → 3.080 (80 ms)** | 2.040 → 2.120 |
| lazy | 2.225 → 2.550 | **3.080 → 3.100 (20 ms)** | 2.120 → 2.320 |
| dog, | 2.600 → 3.300 | 3.100 → 3.488 | 2.320 → 2.740 |
| then | 3.350 → 3.525 | 3.488 → 3.877 | 3.240 → 3.420 |
| runs | 3.575 → 3.775 | 3.877 → 4.720 | 3.420 → 3.600 |
| away | 3.800 → 4.025 | 4.720 → 4.780 | 3.600 → 3.880 |
| fast. | 4.100 → 5.025 | 4.780 → 5.050 | 3.880 → 4.220 |

Acoustic reality: speech only *begins* at 0.390 s (leading silence 0.00-0.39) → the current code puts
the `The|quick` boundary at 0.200 s **inside the leading silence**. Of the three real inter-word pauses
(centres 0.720 s, 1.570 s, 3.115 s) the current heuristic assigns the **correct word pair to 1 of 3**
(the 1.570 s pause is given to `jumps|over`, the 3.115 s pause to `lazy|dog,`). Boundary error vs the
Whisper reference: **mean 524 ms, p50 478 ms, max 1120 ms.**

### Root causes found in `internal/shared/shared.go:78-217` (`SilenceBoundaryTimings`)
1. **Gap centres are not enforced monotonic.** When a found centre falls behind the previous one, the code
   bails out to `end = currentStart + totalDuration/len(boundaries)` — the source of the fake constant
   0.3885 s slices seen above.
2. **Independent local search per gap** (`±15 %` of total duration, "longest silence wins") → pauses get
   attached to the wrong word pair. It must be a single monotonic sequence alignment (DP).
3. **`TokenizeWordBoundaries` token counts do not match the audio.** Per-word phonemization sums to
   **66 tokens** for the reference sentence while `TokenizeWithVoice` (what is actually synthesized)
   produces **77** — the 11 space phonemes are dropped. The docstring's "the total token sequence produced
   here matches `TokenizeWithVoice`" is false, so the proportional prior is systematically skewed.
   One phonemization pass split on the space phoneme gives exactly 13 groups matching the 13 words.
4. Threshold is relative to **peak RMS** (`-35 dB`), and 20 ms windows — a whisper-quiet word breaks it.
5. `?timestamps=1` re-synthesizes the whole utterance (2× CPU for every timed request).

---

## 3. The upgrade that is already proven locally

### 3a. KittenTTS — free, exact, no new assets
`duration` is a first-class graph output of the model zen-tts already ships. Verified
(`kitten_duration_probe.py`, `kitten_zen_parity.py`):

```
graph outputs: ['waveform', 'duration']          # mini AND nano
duration: int64[len(tokens)]   600.00 samples/frame   sum*600 == samples   EXACT
```
Word timings obtained that way (phoneme groups, `luna`/`expr-voice-3-f`, continuous speech,
Whisper reference from the *server's* audio for the same text):

```
'The' 0.475->0.575  'quick' 0.650->0.925  'brown' 0.975->1.275 ... 'fast.' 4.725->5.65
```

### 3b. Kokoro — durations side-graph, frame-exact against production audio
A full-model re-export fails: `torch.onnx.export(KModelForONNX)` → **`RuntimeError: Unknown number
type: complex`** (ISTFTNet complex decoder) — that is why public exports are audio-only. But the
duration predictor exports fine on its own:

```
kokoro_durations_only.onnx   55.9 MB   tokens+style+speed -> durations[int64, seq]
```
Cross-checked against the **live server's** audio (same tokenizer via `piper_phonemize`, same Vocab,
same style row):

| check | result |
|---|---|
| isolated words (13): `sum(durations)*600` vs server sample count | **13 / 13 EXACT** |
| sentences (4): predicted vs server sample count | **EXACT** (121 200 / 123 600 / 107 400 / 129 600) |
| side-graph vs torch reference on identical tokens | **bit-identical** (152 frames both) |

| Kokoro side-graph CPU cost | **39.8 ms**/call vs **903.3 ms** for the audio graph (16 threads) → ~4 % overhead |
| ORT `session.random_seed` on KittenTTS | accepted, but two calls on the **same** session still differ; a **fresh seeded session** is bit-reproducible (cost 426 ms to build) — so intra-session determinism needs graph surgery: the noise lives in `/decoder/generator/m_source/l_sin_gen/RandomUniformLike` and `.../RandomNormalLike` |

Two integration details that must not be missed:
* **Style row**: `internal/kokoro/engine.go:101-111` selects the style embedding **row `len(tokens)`** of a
  510×256 table (voice packs here are 510×256, not hexgrad's single 256-vector). The durations graph must
  be fed the *same row*, otherwise durations are ~30 % off (measured 40 vs 51 frames for `"The"`).
* Text normalization runs first (`normalizer.go` appends `.` when missing) — durations must be computed on
  the **normalized** text, same tokens as the audio call.

---

## 4. Next actions (ordered, with effort)

**P0 — Kitten exact timings (~30 LOC, no assets, do first)**
`internal/kitten/engine.go`: request `[]string{"waveform","duration"}`, keep the existing `hasSpeed`
fallback path; add `SynthesizeWithTimings` so `KittenEngine` satisfies `TimingEngine`; shared helper in
`internal/shared` that (a) phonemizes **once**, (b) groups tokens per word by splitting on the `space`
phoneme, (c) returns `WordTiming{start,end}` via `cumsum(duration)*600/sampleRate`. Also fix the
"timings are zero-width" expectation: emit `phonemes []` next to words (it is free) so downstream
(HyperFrames visemes) can use frame-level data.

**P0 — fix the `?timestamps=1` contract**
One synthesis call: have `TTSEngine.Synthesize` internally return timings (or call only
`SynthesizeWithTimings` and derive PCM from it) — today the same text is synthesized twice
(`server.go:388-402`, `:447`). Consider returning raw PCM + a separate `?timestamps=1` sidecar
instead of base64 (33 % payload inflation).

**P1 — Kokoro exact timings (side-graph)**
Ship `kokoro-v1.0-durations.onnx` (55.9 MB, mostly the ALBERT sub-model; fp16 ≈ 28 MB) next to the audio
model, auto-downloaded by `config.go:downloadIfNeeded`. In `internal/kokoro/engine.go`: same inputs as the
audio call (normalized tokens, same style row, speed) → `durations` → identical word/phoneme timing code as
Kitten. Cost measured: **39.8 ms** per call vs 903.3 ms for the audio graph (16 CPU threads), and it is the
*small* half of the model.
Fallback for users without the file: the improved VAD heuristic (P2).

**P1 — determinism contract**
`duration`/`pred_dur` are stable, waveforms are not: KittenTTS' ONNX carries
`RandomUniformLike` + `RandomNormalLike` inside `decoder/generator/m_source/l_sin_gen`, and ORT's
`session.random_seed` only makes a **freshly created** seeded session reproducible (two calls on one
seeded session still differ; building a session costs 426 ms). Practical options, cheapest first:
(a) graph-surgery pass on the Kitten model — replace those two nodes with `Constant`/graph-input noise
and keep the original as `…-noised.onnx`; (b) per-seed session pool in `KittenEngine` (memory cost, no
model edit); (c) document that only Kokoro (ONNX) is bit-reproducible. For the Python path, document
`torch.manual_seed` — the torch decoder injects `torch.randn_like` (`kokoro/istftnet.py:205,253`).

**P2 — make the VAD fallback honest (keep for Piper / third-party voices)**
In `SilenceBoundaryTimings`: (1) replace per-gap independent search with a **single monotone DP**
(cost = |measured gap duration − predicted token-proportional duration|), (2) forbid non-monotonic
boundaries instead of falling back to equal slices, (3) use `TokenizeWordBoundaries` replaced by the
"one phonemization + space-split grouping" counts from P0 (fixes the 66-vs-77 mismatch), (4) relative
threshold from the p90 speech frame, not peak, and 10 ms windows, (5) clamp first/last boundary to the
measured leading/trailing silence (that alone kills the "boundary at 0.0 in 0.4 s of silence" failure).

**P3 — API + docs**
Document in `TTS_API.md`: `?timestamps=1`, per-word *and* per-phoneme output, 25 ms/12.5 ms quantization,
`word.end` includes the following pause unless `trim_silence=true`, and per-engine support matrix
(Kitten: exact, Kokoro: exact with durations side-car, Piper: VAD only). Add `frame_rate` convenience
field so consumers can do `frame = floor(start*fps)` server-side. Optional `?timestamps=2` → phoneme level.

**Not worth doing:** adding a Python/torch runtime to zen-tts to get `result.tokens` (the C1 path). It
works (measured above) but costs a 330 MB torch process, is non-deterministic in audio, and the ONNX
side-graph gives the same numbers at native speed.

---

## 5. Reproduce

```bash
cd scratch/timing
../pytts/bin/python -c "…"                    # env: uv venv + kokoro==0.9.4 kittentts faster-whisper onnxscript
ZENTTS=http://127.0.0.1:5055 ../pytts/bin/python zen_tts_probe.py          # live server, both engines
../pytts/bin/python py_kokoro_probe.py                                     # C1/C2/C3/C7
../pytts/bin/python reference_align.py                                     # Whisper reference (audio-only)
../pytts/bin/python compare_report.py                                      # 524 ms vs 204 ms table
../pytts/bin/python kitten_duration_probe.py ; ../pytts/bin/python kitten_zen_parity.py   # C5
../pytts/bin/python onnx_export_probe.py                                   # C4 + 3b (B succeeds, A fails)
../pytts/bin/python kokoro_word_timings_poc.py                            # 13/13 + 4/4 EXACT
../pytts/bin/python silence_anchor_table.py                               # anchor table in §2
```
Artifacts: `timing_accuracy.{md,json}`, `whisper_align.json`, `silence_anchor_table.json`,
`kitten_duration_report.json`, `kitten_zen_parity.json`, `kokoro_duration_parity.json`,
`py_kokoro_timings.json`, `zen_probe.json`, `gap_energy_report.json`, `acoustic_report.json`,
`pausal_control.json`, `poc_acoustic_check.json`, `matched_filter_check.json` (envelope matched-filter
reference — unreliable for short function words, kept as a negative result), `kokoro_durations_only.onnx`.
