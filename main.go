package main

import (
	"flag"
	"log"
	"os"
)

var DebugMode bool

func DebugMsg(format string, v ...interface{}) {
	if DebugMode {
		log.Printf("[DEBUG] "+format, v...)
	}
}

func main() {
	modelPtr := flag.String("m", "", "Override voice model")
	portPtr := flag.Int("p", 0, "Override port")
	cpuPtr := flag.Int("c", 0, "Pin to CPU core (e.g. 0)")
	uiPtr := flag.Bool("ui", false, "Show TUI")
	debugPtr := flag.Bool("debug", false, "Enable debug logs")
	flag.Parse()

	DebugMode = *debugPtr

	// Check Binary
	if _, err := os.Stat(PiperBinary); os.IsNotExist(err) {
		log.Fatalf("❌ Critical: Piper binary not found at %s", PiperBinary)
	}

	// Initialization
	LoadConfig()
	LoadRegistry()

	// Apply CLI overrides to config
	if *portPtr != 0 { CurrentConfig.Port = *portPtr }
	if *modelPtr != "" { CurrentConfig.LastModel = *modelPtr }

	if *uiPtr {
		RunTUI(CurrentConfig.Port, *cpuPtr)
	} else {
		// Headless Mode
		StartServer(CurrentConfig.LastModel, CurrentConfig.Port, *cpuPtr)
		select {} // Block forever
	}
}