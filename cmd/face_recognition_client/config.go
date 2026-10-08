package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	defaultWebUIRoute       = "/face-recognition"
	defaultEmbeddingsDBPath = "face_db/embeddings.json"
	defaultModelDir         = "model"
)

type FaceClientConfig struct {
	GravityTCPHost string
	WebUIEnabled   *bool
	WebUIListen    string
	WebUIRoute     string

	CameraDeviceID string
	CameraFPS      int32
	CameraWidth    int32
	CameraHeight   int32

	DetectModelID   string
	DetectModelPath string
	EmbedModelID    string
	EmbedModelPath  string
	UploadModels    bool

	EmbeddingsFile        string
	MatchThreshold        float32
	SinglePersonThreshold float32
	SecondBestMargin      float32
	MaxFaces              int
	CropMargin            float32
	ScoreThreshold        float32
	CameraRetrySeconds    int
	ExecutionMode         string
	RegisterInAppHub      bool
	EnrollTargetSamples   int
	EnrollDurationSeconds float32
}

func hardcodedConfig() FaceClientConfig {
	webUI := true
	return FaceClientConfig{
		GravityTCPHost: "192.168.1.100", // development only; on the device the SDK connects locally
		WebUIEnabled:   &webUI,
		WebUIListen:    "", // automatic: 127.0.0.1 and a port of the reserved range (see webui.go)
		WebUIRoute:     defaultWebUIRoute,

		CameraDeviceID: "/dev/video0",
		CameraFPS:      15,
		CameraWidth:    640,
		CameraHeight:   480,

		DetectModelID:   "blazeface_full",
		DetectModelPath: filepath.Join(defaultModelDir, "blaze_face_full_range.tflite"),
		EmbedModelID:    "sface",
		EmbedModelPath:  filepath.Join(defaultModelDir, "face_recognition_sface_2021dec_int8.onnx"),
		UploadModels:    true,

		EmbeddingsFile: defaultEmbeddingsDBPath,
		// Similarities given by SFace: about 0.65 on average for two pictures of
		// the same person, about 0.1 for different people (see the README).
		MatchThreshold:        0.40,
		SinglePersonThreshold: 0.45,
		SecondBestMargin:      0.08,
		MaxFaces:              2,
		CropMargin:            0.25,
		ScoreThreshold:        0.5,
		CameraRetrySeconds:    15,
		ExecutionMode:         "cpu",
		RegisterInAppHub:      true,
		EnrollTargetSamples:   8,
		EnrollDurationSeconds: 5.0,
	}
}

func loadFaceClientConfig() (FaceClientConfig, error) {
	cfg := hardcodedConfig()
	applyConfigEnvOverrides(&cfg)
	resolveModelPaths(&cfg)
	if err := validateFaceClientConfig(&cfg); err != nil {
		return FaceClientConfig{}, err
	}
	return cfg, nil
}

func resolveModelPaths(cfg *FaceClientConfig) {
	if cfg == nil {
		return
	}
	cfg.DetectModelPath = resolveModelPath(cfg.DetectModelPath)
	cfg.EmbedModelPath = resolveModelPath(cfg.EmbedModelPath)
}

func resolveModelPath(modelPath string) string {
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return modelPath
	}
	base := filepath.Base(modelPath)
	candidates := []string{
		base,
		filepath.Join("data", base),
		modelPath,
		filepath.Join(defaultModelDir, base),
		filepath.Join("orb", "data", base),
		filepath.Join("cmd", "face_recognition_client", defaultModelDir, base),
		filepath.Join("cmd", "face_recognition_client", "orb", "data", base),
	}
	for _, root := range candidateRoots() {
		for _, rel := range candidates {
			path := filepath.Join(root, rel)
			if fileExists(path) {
				return path
			}
		}
	}
	return modelPath
}

func candidateRoots() []string {
	roots := make([]string, 0, 8)
	seen := map[string]bool{}
	add := func(root string) {
		root = strings.TrimSpace(root)
		if root == "" {
			return
		}
		root = filepath.Clean(root)
		if seen[root] {
			return
		}
		seen[root] = true
		roots = append(roots, root)
	}
	if cwd, err := os.Getwd(); err == nil {
		for d := cwd; ; {
			add(d)
			parent := filepath.Dir(d)
			if parent == d {
				break
			}
			d = parent
		}
	}
	if exe, err := os.Executable(); err == nil {
		for d := filepath.Dir(exe); ; {
			add(d)
			parent := filepath.Dir(d)
			if parent == d {
				break
			}
			d = parent
		}
	}
	return roots
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir()
}

func applyConfigEnvOverrides(cfg *FaceClientConfig) {
	if cfg == nil {
		return
	}
	if v := strings.TrimSpace(os.Getenv("ORBIT_GRAVITY_TCP_HOST")); v != "" {
		cfg.GravityTCPHost = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_CAMERA_DEVICE_ID")); v != "" {
		cfg.CameraDeviceID = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_CAMERA_FPS")); v != "" {
		fmt.Sscanf(v, "%d", &cfg.CameraFPS)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_CAMERA_WIDTH")); v != "" {
		fmt.Sscanf(v, "%d", &cfg.CameraWidth)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_CAMERA_HEIGHT")); v != "" {
		fmt.Sscanf(v, "%d", &cfg.CameraHeight)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_MAX_FACES")); v != "" {
		fmt.Sscanf(v, "%d", &cfg.MaxFaces)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_DETECT_MODEL_ID")); v != "" {
		cfg.DetectModelID = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_DETECT_MODEL_PATH")); v != "" {
		cfg.DetectModelPath = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_EMBED_MODEL_ID")); v != "" {
		cfg.EmbedModelID = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_EMBED_MODEL_PATH")); v != "" {
		cfg.EmbedModelPath = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_WEBUI_LISTEN")); v != "" {
		cfg.WebUIListen = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_WEBUI_ROUTE")); v != "" {
		cfg.WebUIRoute = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_EMBEDDINGS_FILE")); v != "" {
		cfg.EmbeddingsFile = v
	}
	if v := strings.TrimSpace(os.Getenv("FACE_MODE_MATCH_THRESHOLD")); v != "" {
		fmt.Sscanf(v, "%f", &cfg.MatchThreshold)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_MATCH_THRESHOLD")); v != "" {
		fmt.Sscanf(v, "%f", &cfg.MatchThreshold)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_MODE_SINGLE_THRESHOLD")); v != "" {
		fmt.Sscanf(v, "%f", &cfg.SinglePersonThreshold)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_SINGLE_PERSON_THRESHOLD")); v != "" {
		fmt.Sscanf(v, "%f", &cfg.SinglePersonThreshold)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_MODE_MARGIN_THRESHOLD")); v != "" {
		fmt.Sscanf(v, "%f", &cfg.SecondBestMargin)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_SECOND_BEST_MARGIN")); v != "" {
		fmt.Sscanf(v, "%f", &cfg.SecondBestMargin)
	}
	if v := strings.TrimSpace(os.Getenv("FACE_ENROLL_DURATION_SECONDS")); v != "" {
		fmt.Sscanf(v, "%f", &cfg.EnrollDurationSeconds)
	}
}

func validateFaceClientConfig(cfg *FaceClientConfig) error {
	if cfg.CameraFPS < 1 {
		return fmt.Errorf("camera_fps must be >= 1")
	}
	if cfg.CameraWidth < 1 || cfg.CameraHeight < 1 {
		return fmt.Errorf("camera_width and camera_height must be >= 1")
	}
	if strings.TrimSpace(cfg.DetectModelID) == "" || strings.TrimSpace(cfg.DetectModelPath) == "" {
		return fmt.Errorf("detect model id/path must be set")
	}
	if strings.TrimSpace(cfg.EmbedModelID) == "" || strings.TrimSpace(cfg.EmbedModelPath) == "" {
		return fmt.Errorf("embed model id/path must be set")
	}
	if strings.TrimSpace(cfg.EmbeddingsFile) == "" {
		return fmt.Errorf("embeddings_file must be set")
	}
	if cfg.MatchThreshold < 0 || cfg.MatchThreshold > 1 {
		return fmt.Errorf("match_threshold must be between 0 and 1")
	}
	if cfg.SinglePersonThreshold < 0 || cfg.SinglePersonThreshold > 1 {
		return fmt.Errorf("single_person_threshold must be between 0 and 1")
	}
	if cfg.SecondBestMargin < 0 || cfg.SecondBestMargin > 1 {
		return fmt.Errorf("second_best_margin must be between 0 and 1")
	}
	if cfg.MaxFaces < 1 {
		return fmt.Errorf("max_faces must be >= 1")
	}
	if cfg.CropMargin < 0 || cfg.CropMargin > 1 {
		return fmt.Errorf("crop_margin must be between 0 and 1")
	}
	if cfg.ScoreThreshold < 0 || cfg.ScoreThreshold > 1 {
		return fmt.Errorf("score_threshold must be between 0 and 1")
	}
	if cfg.CameraRetrySeconds < 1 {
		return fmt.Errorf("camera_retry_seconds must be >= 1")
	}
	if cfg.EnrollTargetSamples < 1 {
		return fmt.Errorf("enroll_target_samples must be >= 1")
	}
	if cfg.EnrollDurationSeconds < 0.5 {
		return fmt.Errorf("enroll_duration_seconds must be >= 0.5")
	}
	mode := strings.ToLower(strings.TrimSpace(cfg.ExecutionMode))
	if mode != "cpu" && mode != "gpu" && mode != "high_threads" {
		return fmt.Errorf("execution_mode must be cpu, gpu, or high_threads")
	}
	return nil
}

func webUIEnabled(cfg *FaceClientConfig) bool {
	if cfg == nil || cfg.WebUIEnabled == nil {
		return true
	}
	return *cfg.WebUIEnabled
}
