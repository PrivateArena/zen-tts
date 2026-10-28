Yes, local non-autoregressive architectures like **Kokoro** (and other StyleTTS2 / VITS derivatives) expose or generate exact speech alignment timing, which makes them very well-suited for frame-accurate animation synchronization.

Here is how speech timing works in Kokoro and how it connects to animation engines like HyperFrames:

---

### 1. Does Kokoro Expose Speech Timing natively?

**Yes, at the Python runtime level.**

Because Kokoro is based on a non-autoregressive architecture (StyleTTS2 with ISTFTNet vocoder), the text is converted into phonemes via a G2P (Grapheme-to-Phoneme) engine, and the internal duration predictor computes duration steps for each token/phoneme before audio synthesis happens.

* **Python (`kokoro` package):**
When running inference in Python via `KPipeline`, the returned output object contains a `result.tokens` array. Each token object provides explicit time boundary metadata:
```json
[
  {"text": "Hello", "start_ts": 0.00, "end_ts": 0.38},
  {"text": "world", "start_ts": 0.42, "end_ts": 0.85}
]

```


This gives you word/token-level start and end timestamps directly out of the box without requiring an external post-inference alignment step.
* **JS/ONNX Web Runtimes (`kokoro-js`):**
Standard web ONNX exports typically output raw PCM float arrays without returning the model's internal intermediate duration hidden states. In browser runtimes, timing is commonly derived by tracking chunk duration over streamed sentences or running a lightweight forced aligner.

---

### 2. Synchronization with Animation (HyperFrames)

When syncing speech with animation (whether for mouth visemes, caption highlighting, or canvas motion graphs), synchronization relies on **phoneme/word timestamps mapped to frame numbers**.

#### Frame Alignment Math

If HyperFrames renders at $N$ FPS (e.g., 30 FPS or 60 FPS), you map the token/phoneme seconds ($t_{\text{seconds}}$) to exact frame indices:

$$\text{frame\_index} = \lfloor t_{\text{seconds}} \times \text{FPS} \rfloor$$

For example:

* A token starting at `0.42s` rendered at `60 FPS` starts on frame $\lfloor 0.42 \times 60 \rfloor = 25$.

#### How HyperFrames Uses This:

1. **Motion Drivers:** HyperFrames uses the timing metadata array returned during Kokoro's inference step to sequence scene elements, triggers, captions, and character visemes directly along the visual timeline.
2. **Determinism:** Because Kokoro computes fixed duration outputs per phoneme for a given random seed, the audio buffer and timing metadata array are deterministic and locked together before frame composition begins.

---

### 3. Overview: Timing Support across Local TTS Models

| TTS Model | Architecture Type | Exposes Timing directly? | Timing Granularity | Best Use Case for Animation |
| --- | --- | --- | --- | --- |
| **Kokoro-82M** | StyleTTS2 / Duration Predictor | **Yes** (in Python via `result.tokens`) | Word / Token | Ideal for low-latency video generation pipelines. |
| **KittenTTS** | Compact FastSpeech2 variant | **Yes** (Duration predictor output) | Phoneme / Frame | Direct viseme lip-sync mapping. |
| **Piper TTS** | VITS (Monotonic Alignment Search) | **Yes** (Phoneme durations via ONNX/C++) | Phoneme / Millisecond | Real-time interactive UI avatar animation. |
| **Bark / XTTS** | Autoregressive (Audio LM) | **No** (Sample-by-sample output) | None (Requires Whisper / MFA) | Requires secondary forced alignment (e.g. CTC / WhisperX). |

### Conclusion

If you generate audio with Kokoro in Python, you get token timing metadata directly. By parsing `result.tokens`, HyperFrames converts those timestamp offsets directly into keyframe triggers to maintain lockstep sync between speech and visual animations.