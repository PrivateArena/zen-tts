package main

// --- FILE PATH & DIRECTORY CONSTANTS ---
const (
	VoicesFile       = "voices.json"
	ConfigFile       = "config.json"
	ReplacementsFile = "replacements.txt"
	PiperDir         = "./piper"
	PiperBinary      = "./piper/piper"
	ModelDir         = "./models"
)

// --- DOWNLOAD URLS ---
const (
	// Piper
	PiperDownloadBase = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"

	// Kokoro-82M
	KokoroOnnxURL  = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"
	KokoroConfURL  = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/config.json"
	KokoroBellaURL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_bella.bin"

	// KittenTTS
	KittenVoicesURL   = "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/voices.npz"
	KittenMiniOnnxURL = "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/kitten_tts_mini_v0_8.onnx"
	KittenMiniConfURL = "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/config.json"

	KittenMicroOnnxURL = "https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/kitten_tts_micro_v0_8.onnx"
	KittenMicroConfURL = "https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/config.json"

	KittenNanoOnnxURL = "https://huggingface.co/KittenML/kitten-tts-nano-0.8/resolve/main/kitten_tts_nano_v0_8.onnx"
	KittenNanoConfURL = "https://huggingface.co/KittenML/kitten-tts-nano-0.8/resolve/main/config.json"
)
