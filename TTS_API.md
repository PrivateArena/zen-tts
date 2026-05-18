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

#### Payload Example
```json
{
  "text": "The quick brown fox jumps over the lazy dog.",
  "voice": "af_bella",
  "speed": 1.0
}
```

#### Response Details
- **Success (200 OK)**: Returns the raw binary `audio/wav` stream (16-bit PCM, mono, 24kHz or 22kHz depending on active engine model).
- **Error (400 Bad Request)**: Empty text or malformed JSON payload.
- **Error (405 Method Not Allowed)**: Triggered for GET or HEAD requests.
- **Error (500 Internal Server Error)**: Engine not initialized or generation failure.

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
  http://127.0.0.1:5000/tts \
  -o output.wav
```

### 2. Python
```python
import requests

url = "http://127.0.0.1:5000/tts"
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

fetch('http://127.0.0.1:5000/tts', {
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

	resp, err := http.Post("http://127.0.0.1:5000/tts", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	file, _ := os.Create("go_output.wav")
	defer file.Close()

	io.Copy(file, resp.Body)
}
```
