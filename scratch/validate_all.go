//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"zen-tts/internal/kokoro"
	"zen-tts/internal/shared"
)

type TestCase struct {
	Language string
	Voice    string
	Text     string
}

func main() {
	fmt.Println("--- INITIALIZING ONNX RUNTIME ---")
	err := shared.InitONNXRuntime()
	if err != nil {
		fmt.Printf("Failed to initialize ONNX Runtime: %v\n", err)
		os.Exit(1)
	}

	modelPath := "./models/kokoro/kokoro-v0.19.onnx"
	fmt.Printf("--- INITIALIZING KOKORO ENGINE (%s) ---\n", modelPath)
	engine := &kokoro.KokoroEngine{}
	if err := engine.Initialize(modelPath, "./models/kokoro/kokoro-v0.19.json"); err != nil {
		fmt.Printf("Failed to initialize Kokoro engine: %v\n", err)
		os.Exit(1)
	}

	testCases := []TestCase{
		{"American English (en-us)", "af_bella", "Hello, this is a test of American English speech synthesis."},
		{"British English (en-gb)", "bf_emma", "Hello, this is a test of British English speech synthesis."},
		{"Japanese (ja)", "jf_nezumi", "こんにちは、日本語の音声合成テストです。"},
		{"Chinese (zh)", "zf_xiaoxiao", "你好，这是中文语音合成的测试。"},
		{"Spanish (es)", "ef_dora", "Hola, esta es una prueba de síntesis de voz en español."},
		{"French (fr)", "ff_siwis", "Bonjour, ceci est un test de synthèse vocale en français."},
		{"Hindi (hi)", "hf_alpha", "नमस्ते, यह हिंदी भाषण संश्लेषण का एक परीक्षण है।"},
		{"Italian (it)", "if_sara", "Ciao, questo è un test di sintesi vocale in italiano."},
		{"Brazilian Portuguese (pt-br)", "pf_dora", "Olá, este é um teste de síntese de voz em português brasileiro."},
	}

	failed := false
	for _, tc := range testCases {
		fmt.Printf("\n--- TESTING LANGUAGE: %s (%s) ---\n", tc.Language, tc.Voice)
		fmt.Printf("Text: %s\n", tc.Text)

		samples, _, err := engine.Synthesize(tc.Text, tc.Voice, 1.0)
		if err != nil {
			fmt.Printf("❌ FAILED: %v\n", err)
			failed = true
			continue
		}

		if len(samples) == 0 {
			fmt.Println("❌ FAILED: returned empty audio samples")
			failed = true
			continue
		}

		// Calculate statistics
		var maxVal float32 = 0
		var minVal float32 = 0
		var sumAbs float32 = 0
		for _, s := range samples {
			if s > maxVal {
				maxVal = s
			}
			if s < minVal {
				minVal = s
			}
			if s < 0 {
				sumAbs -= s
			} else {
				sumAbs += s
			}
		}
		meanAbs := sumAbs / float32(len(samples))

		fmt.Printf("✅ SUCCESS: generated %d samples (approx %.2f seconds)\n", len(samples), float32(len(samples))/24000.0)
		fmt.Printf("Statistics -> Max: %.4f, Min: %.4f, Mean Absolute: %.4f\n", maxVal, minVal, meanAbs)
	}

	// Clean up downloaded cache files so we don't pollute git
	fmt.Println("\n--- CLEANING UP DOWNLOADED CACHES ---")
	voicesDir := "./models/kokoro/voices"
	for _, tc := range testCases {
		binPath := filepath.Join(voicesDir, tc.Voice+".bin")
		if tc.Voice != "af_bella" && tc.Voice != "af_jasper" {
			if _, err := os.Stat(binPath); err == nil {
				os.Remove(binPath)
				fmt.Printf("Removed temporary cache: %s\n", binPath)
			}
		}
	}

	if failed {
		fmt.Println("\n❌ Test execution completed with failures.")
		os.Exit(1)
	} else {
		fmt.Println("\n🎉 All 9 languages tested and verified successfully!")
	}
}
