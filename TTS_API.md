# Zen-TTS API Reference

Welcome to the **Zen-TTS** backend API documentation. The server provides a high-performance, real-time Text-to-Speech (TTS) engine supporting dynamic switching between Piper, Kokoro-82M, and KittenTTS models.

---

## Endpoint Overview

### `POST /tts`
Synthesizes speech from the provided text and returns a binary `audio/wav` file.

#### Headers
- `Content-Type: application/json`
- `Access-Control-Allow-Origin: *` (CORS support enabled)

#### Request Payload
The request body must be a JSON object with the following fields:

| Field | Type | Required | Default | Description |
| :--- | :--- | :---: | :---: | :--- |
| `text` | `string` | **Yes** | — | The sentence/text to synthesize. Automatic text normalization is applied. |
| `voice` | `string` | **Yes** | — | Voice model key matching the active model's voice vocabulary. |
| `speed` | `float` | No | `1.0` | Speaking speed multiplier. Must be greater than `0.0`. |
| `timestamps` | `int` | No | `0` | `1` = per-word, `2` = per-word + per-phoneme JSON envelope. Equivalent to `?timestamps=` on the URL, which wins. |
| `fps` | `float` | No | `30` | Animation frame rate echoed back as `frame_rate`. |

#### Payload Example
```json
{
  "text": "The quick brown fox jumps over the lazy dog.",
  "voice": "af_bella",
  "speed": 1.0
}
```

#### Response Details
- **Success (200 OK)**: Returns the raw binary `audio/wav` stream (16-bit PCM, mono, 24kHz or 22kHz depending on active engine model). With `?timestamps=` the response is JSON instead — see [Speech Timing](#speech-timing-word--phoneme-level) below.
- **Error (400 Bad Request)**: Empty text or malformed JSON payload.
- **Error (405 Method Not Allowed)**: Triggered for GET or HEAD requests.
- **Error (500 Internal Server Error)**: Engine not initialized or generation failure.

---

## Speech Timing (word & phoneme level)

Add `?timestamps=1` (words) or `?timestamps=2` (words + phonemes) — or the JSON body field
`"timestamps": 1|2` (the URL parameter wins) — to get the alignment of the utterance next
to its audio. **The audio and the timings always come out of one synthesis pass**, so the
timings can never describe a different rendering than the WAV you received.

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"text": "The quick brown fox jumps over the lazy dog.", "voice": "af_bella"}' \
  'http://127.0.0.1:5055/tts?timestamps=2&fps=30' -o timing.json
```

### Response Body

| Field | Type | Notes |
| :--- | :--- | :--- |
| `audio` | `string` | base64 of the same WAV the plain request returns (16-bit PCM mono). |
| `timings` | `array\|null` | Per word: `{word, start, end}` in seconds from segment start. `null` when the active engine cannot align (see the matrix). |
| `duration` | `float` | Audio length in seconds. |
| `sample_rate` | `int` | Sample rate of `audio`. |
| `frame_rate` | `float` | Echo of `?fps=` / `"fps"`, default `30`. See frame mapping below. |
| `frame_ms` | `float` | Resolution of the numbers: `25` for model frames, `10` for estimates. |
| `timing_source` | `string` | `duration_graph` (exact) · `vad_estimate` (silence analysis) · `estimated` (engine did not say) · `none`. |
| `phonemes` | `array` | Only with level `2`, and only for engines with a duration output: `{phoneme, word_index, start, end, frames}`. |

### Frame mapping

`frame_index = floor(t_seconds * frame_rate)`

Boundaries are multiples of `frame_ms`, so at 30 FPS (33.3 ms/frame) they are at most one
frame off, and at 60 FPS half a frame. Word starts are the reliable edge; a word's `end`
runs to the middle of the following pause, so the pause is *included* in the word. The
closing `EOS` pause belongs to no word: the last word ends before `duration`.

### Per-engine support matrix

| Engine | `timing_source` | Granularity | Where it comes from |
| :--- | :--- | :--- | :--- |
| KittenTTS (`kitten-tts-mini/micro/nano`) | `duration_graph` | word + phoneme | The ONNX graph returns `duration` next to `waveform`: one frame count per token, `sum(duration) × 600 ==` samples. |
| Kokoro-82M (with side-car) | `duration_graph` | word + phoneme | `models/kokoro/kokoro-v1.0-durations.onnx`, run with the same tokens, style row and speed as the audio graph. |
| Kokoro-82M (without side-car) | `vad_estimate` | word | Silence analysis of the produced PCM. |
| Piper | `vad_estimate` | word | VITS durations are training-time only; the shipped ONNX exports return audio. |
| Inflect-Micro-v2 | `none` | — | No alignment source yet; `timings` is `null` and the audio is unchanged. |

`?timestamps=1` never fails because of a missing timing source: you get the audio and
`"timings": null`, and `timing_source` tells you so.

#### Installing the Kokoro duration side-car

The public Kokoro ONNX export is audio-only, so the duration predictor ships as a small
side-graph (≈ 56 MB, ~4 % of the audio graph's cost). The export from
`hexgrad/Kokoro-82M` needs one fix before this process's ONNX Runtime will load it: its
`TopK` nodes take `K` from a `Shape → Gather` chain, which older runtimes reject during
shape inference (`K` is always `[1]` here). The export plus the patch are scripted:

```bash
scratch/pytts/bin/python scratch/timing/patch_durations_for_old_ort.py \
    scratch/timing/kokoro_durations_only.onnx \
    models/kokoro/kokoro-v1.0-durations.onnx
```

The engine picks the file up automatically from `<model>-durations.onnx` and falls back to
`vad_estimate` when it is absent. Verify it is in use by checking that
`timing_source == "duration_graph"` and that `sum` of a word's phoneme `frames × 600` equals
its sample span.

### Determinism

`?timestamps` output is reproducible, the *waveform* is not always so:

| Path | Bit-reproducible audio? | Timings stable? |
| :--- | :--- | :--- |
| Kokoro (ONNX, this server) | yes | yes |
| KittenTTS (ONNX) | **no** — the decoder contains `RandomUniformLike`/`RandomNormalLike` sources | yes (the `duration` tensor is fixed) |
| Kokoro (Python `KPipeline`) | no, `istftnet` injects `randn_like` unless `torch.manual_seed` is set | yes (`pred_dur` is fixed) |

So a re-rendered Kitten clip keeps its alignment (word/phoneme frames) but not its exact
samples: for frame-locked animation, cache the WAV together with its timings in one request
rather than synthesizing twice.


---

## Active Voice Models & Keys

Depending on the engine model you booted the server with (`-m` flag), use the corresponding `voice` parameter keys:

### 1. Kokoro-82M Engine (`-m kokoro-v1.0`)
- **Default Sample Rate**: 24,000 Hz
- **Voice Keys**:
  - `af_bella` (Female, Realism)
  - `af_sarah` (Female, Standard)
  - `am_adam` (Male, Standard)
  - `am_michael` (Male, Standard)
  - *(And other standard Kokoro voice codes)*

### 2. KittenTTS Engine (`-m kitten-tts-mini`)
- **Default Sample Rate**: 24,000 Hz
- **Voice Keys**:
  - `Bella` (Female)
  - `Sarah` (Female)
  - `Michael` (Male)

### 3. Piper Engine (e.g. `-m en_US-ryan-medium`)
- **Default Sample Rate**: 22,050 Hz
- **Voice Keys**:
  - `0` (Default / first speaker)
  - *(or corresponding index strings for multi-speaker models)*

---

## Code Integration Examples

### 1. cURL / Shell
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice": "af_bella", "speed": 1.0}' \
  http://127.0.0.1:5055/tts \
  -o output.wav
```

**Tips**: Use /tts?play=true for streaming

### 2. Python
```python
import requests

url = "http://127.0.0.1:5055/tts"
payload = {
    "text": "Hello, this is a python client request.",
    "voice": "af_bella",
    "speed": 1.0
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    with open("python_output.wav", "wb") as f:
        f.write(response.content)
    print("Success! Audio saved as python_output.wav")
else:
    print(f"Error {response.status_code}: {response.text}")
```

### 3. Node.js (JavaScript / Fetch)
```javascript
const fs = require('fs');

const payload = {
  text: "Hello, this is a Node.js client request.",
  voice: "af_bella",
  speed: 1.0
};

fetch('http://127.0.0.1:5055/tts', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
})
.then(res => {
  if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
  return res.arrayBuffer();
})
.then(buffer => {
  fs.writeFileSync('node_output.wav', Buffer.from(buffer));
  console.log('Success! Audio saved as node_output.wav');
})
.catch(err => console.error('Error:', err));
```

### 4. Go
```go
package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
)

type TTSRequest struct {
	Text  string  `json:"text"`
	Voice string  `json:"voice"`
	Speed float64 `json:"speed"`
}

func main() {
	reqBody := TTSRequest{
		Text:  "Hello, this is a Go client request.",
		Voice: "af_bella",
		Speed: 1.0,
	}
	jsonData, _ := json.Marshal(reqBody)

	resp, err := http.Post("http://127.0.0.1:5055/tts", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	file, _ := os.Create("go_output.wav")
	defer file.Close()

	io.Copy(file, resp.Body)
}
```
