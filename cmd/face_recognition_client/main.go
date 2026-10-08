package main

import (
	_ "embed"
	"os"

	"github.com/OrbitOS-org/orbit-os-sdk-go/v26/logger"
	"github.com/OrbitOS-org/orbit-os-sdk-go/v26/metadata"
)

const logTag = "face_recognition_client"

//go:embed metadata.json
var metadataJSON []byte

var appManifest = metadata.MustParseAppManifestJSON(metadataJSON)

func main() {
	logger.Init(appManifest.Name, "INFO", true)
	appManifest.PrintInfo()

	client := NewFaceClient()
	if err := client.Init(); err != nil {
		logger.Fatalf(logTag, "init failed: %v", err)
		os.Exit(1)
	}

	client.Run()
}
