//go:build ignore

// probe_durations: checks whether the loaded Kokoro ONNX model exposes a
// "durations" output tensor alongside "audio". Run from the zen-tts root:
//
//	go run scratch/probe_durations/main.go
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"zen-tts/internal/shared"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	modelPath := filepath.Join("models", "kokoro", "kokoro-v1.0.onnx")
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Model not found at %s\n", modelPath)
		os.Exit(1)
	}

	if err := shared.InitONNXRuntime(); err != nil {
		fmt.Fprintf(os.Stderr, "ONNX init failed: %v\n", err)
		os.Exit(1)
	}

	// --- Probe 1: does ["audio","durations"] work? ---
	fmt.Println("=== Probe 1: requesting [\"audio\",\"durations\"] ===")
	probe1(modelPath)

	// --- Probe 2: list all available output names via model metadata ---
	fmt.Println("\n=== Probe 2: list model output node names ===")
	probeOutputNames(modelPath)
}

func probe1(modelPath string) {
	tokens := []int64{0, 69, 102, 123, 51, 16, 65, 47, 54, 54, 57, 0} // "hello world" roughly
	styleVec := make([]float32, 256)
	speed := []float32{1.0}

	tokShape := ort.NewShape(1, int64(len(tokens)))
	tokTensor, err := ort.NewTensor(tokShape, tokens)
	if err != nil {
		fmt.Printf("  token tensor error: %v\n", err)
		return
	}
	defer tokTensor.Destroy()

	stShape := ort.NewShape(1, 256)
	stTensor, err := ort.NewTensor(stShape, styleVec)
	if err != nil {
		fmt.Printf("  style tensor error: %v\n", err)
		return
	}
	defer stTensor.Destroy()

	spShape := ort.NewShape(1)
	spTensor, err := ort.NewTensor(spShape, speed)
	if err != nil {
		fmt.Printf("  speed tensor error: %v\n", err)
		return
	}
	defer spTensor.Destroy()

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"tokens", "style", "speed"},
		[]string{"audio", "durations"},
		nil)
	if err != nil {
		fmt.Printf("  FAILED to create session with \"durations\": %v\n", err)
		fmt.Println("  → Model does NOT expose a \"durations\" output.")
		return
	}
	defer session.Destroy()

	outputs := []ort.Value{nil, nil}
	err = session.Run([]ort.Value{tokTensor, stTensor, spTensor}, outputs)
	if err != nil {
		fmt.Printf("  session.Run failed: %v\n", err)
		return
	}
	if outputs[0] != nil {
		defer outputs[0].Destroy()
	}
	if outputs[1] != nil {
		defer outputs[1].Destroy()
		durTensor, ok := outputs[1].(*ort.Tensor[float32])
		if ok {
			data := durTensor.GetData()
			fmt.Printf("  ✓ \"durations\" tensor available! Shape has %d elements\n", len(data))
			fmt.Printf("  First 10 values: %v\n", clip(data, 10))
		} else {
			fmt.Printf("  \"durations\" output present but unexpected type: %T\n", outputs[1])
		}
	} else {
		fmt.Println("  \"durations\" output is nil — not exposed by this model export.")
	}
}

func probeOutputNames(modelPath string) {
	// Try requesting a deliberately wrong output name to get the error message
	// which typically lists available names.
	tokens := []int64{0, 43, 0}
	styleVec := make([]float32, 256)
	speed := []float32{1.0}

	tokShape := ort.NewShape(1, int64(len(tokens)))
	tokTensor, _ := ort.NewTensor(tokShape, tokens)
	defer tokTensor.Destroy()
	stShape := ort.NewShape(1, 256)
	stTensor, _ := ort.NewTensor(stShape, styleVec)
	defer stTensor.Destroy()
	spShape := ort.NewShape(1)
	spTensor, _ := ort.NewTensor(spShape, speed)
	defer spTensor.Destroy()

	_, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"tokens", "style", "speed"},
		[]string{"__probe_nonexistent__"},
		nil)
	if err != nil {
		fmt.Printf("  Error with bogus output (should list valid names): %v\n", err)
	}
}

func clip[T any](s []T, n int) []T {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
