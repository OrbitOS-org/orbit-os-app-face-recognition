package main

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/OrbitOS-org/orbit-os-sdk-go/v26/logger"
)

//go:embed web/static/*
var webAssets embed.FS

const webTag = "face_webui"

// Web interface ports are reserved to this range. The app starts at
// preferredPort and moves to the next one when a port is taken or refused.
const (
	portMin       = 50000
	portMax       = 60000
	preferredPort = 50004
)

// startWebUI serves the page and registers it with the Orbit OS Launcher. The
// page listens on this device only (127.0.0.1) and is reached through the
// Launcher, behind the device login. An error means the page could not be
// opened: the app has nothing to offer without it and must stop.
func startWebUI(client *FaceClient) error {
	if !webUIEnabled(&client.cfg) {
		return nil
	}

	staticFS, err := fs.Sub(webAssets, "web/static")
	if err != nil {
		return fmt.Errorf("load the page files: %w", err)
	}
	page, err := stampedPage(staticFS)
	if err != nil {
		return fmt.Errorf("load the page: %w", err)
	}
	static := http.FileServer(http.FS(staticFS))

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" || r.URL.Path == "/index.html" {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cache-Control", "no-cache") // the page itself is always asked for again
			_, _ = w.Write(page)
			return
		}
		w.Header().Set("Cache-Control", "public, max-age=86400")
		static.ServeHTTP(w, r)
	})
	mux.HandleFunc("/api/status", client.handleStatus)
	mux.HandleFunc("/api/results", client.handleResults)
	mux.HandleFunc("/api/mode", client.handleMode)
	mux.HandleFunc("/api/reload-db", client.handleReloadDB)
	mux.HandleFunc("/api/enroll/start", client.handleEnrollStart)
	mux.HandleFunc("/api/enroll/capture", client.handleEnrollCapture)
	mux.HandleFunc("/api/enroll/commit", client.handleEnrollCommit)
	mux.HandleFunc("/api/enroll/cancel", client.handleEnrollCancel)
	mux.HandleFunc("/api/people", client.handlePeople)
	mux.HandleFunc("/api/remove", client.handleRemove)
	mux.HandleFunc("/api/settings", client.handleSettings)
	mux.HandleFunc("/api/camera/select", client.handleCameraSelect)
	mux.HandleFunc("/video_feed", client.handleVideoFeed)

	root := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The models are loaded when the page is first used, not at start.
		_ = client.ensureModelsWarm()
		mux.ServeHTTP(w, r)
	})

	listener, registered, err := listenAndRegister(client)
	if err != nil {
		return err
	}
	client.webRegistered = registered
	if registered {
		logger.Infof(webTag, "page on %s, registered with the Launcher at %s", listener.Addr(), client.webRoute())
	} else {
		logger.Infof(webTag, "page on http://%s (not registered with the Launcher)", listener.Addr())
	}
	go func() {
		if err := http.Serve(listener, root); err != nil {
			logger.Errorf(webTag, "the page stopped: %v", err)
			os.Exit(1) // without its page the app is of no use; Orbit OS starts it again
		}
	}()
	return nil
}

func (c *FaceClient) webRoute() string {
	if route := strings.TrimSpace(c.cfg.WebUIRoute); route != "" {
		return route
	}
	return defaultWebUIRoute
}

// listenAndRegister opens the page's port and registers it with the Launcher.
// With FACE_WEBUI_LISTEN set (development) it listens exactly there.
// Otherwise it looks for a free port in the reserved range, on 127.0.0.1.
func listenAndRegister(client *FaceClient) (listener net.Listener, registered bool, err error) {
	hub := client.cfg.RegisterInAppHub && client.gravity != nil && client.gravity.AppHubManager != nil
	route := client.webRoute()

	if fixed := strings.TrimSpace(client.cfg.WebUIListen); fixed != "" {
		listener, err = net.Listen("tcp", fixed)
		if err != nil {
			return nil, false, fmt.Errorf("listen on %s: %w", fixed, err)
		}
		if hub {
			if err := client.gravity.AppHubManager.RegisterWebUI(listener.Addr().String(), route); err != nil {
				logger.Warnf(webTag, "RegisterWebUI %s: %v", listener.Addr(), err)
				return listener, false, nil
			}
		}
		return listener, hub, nil
	}

	span := portMax - portMin + 1
	for i := 0; i < span; i++ {
		addr := fmt.Sprintf("127.0.0.1:%d", portMin+(preferredPort-portMin+i)%span)
		listener, err = net.Listen("tcp", addr)
		if err != nil {
			continue // port taken: try the next one
		}
		if !hub {
			return listener, false, nil
		}
		if err := client.gravity.AppHubManager.RegisterWebUI(addr, route); err != nil {
			listener.Close() // the Launcher refused this port: try the next one
			logger.Warnf(webTag, "RegisterWebUI %s: %v", addr, err)
			continue
		}
		return listener, true, nil
	}
	return nil, false, fmt.Errorf("no free web interface port in %d-%d", portMin, portMax)
}

// stampedPage returns the page with the app version and with a mark of this
// build in the address of the files it loads. Browsers keep those files for a
// while; with the mark, a new version of the app is never shown with the files
// of the previous one.
func stampedPage(static fs.FS) ([]byte, error) {
	page, err := fs.ReadFile(static, "index.html")
	if err != nil {
		return nil, err
	}
	h := sha256.New()
	err = fs.WalkDir(static, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		data, err := fs.ReadFile(static, path)
		h.Write(data)
		return err
	})
	if err != nil {
		return nil, err
	}
	build := hex.EncodeToString(h.Sum(nil))[:10]
	page = bytes.ReplaceAll(page, []byte("__BUILD__"), []byte(build))
	return bytes.ReplaceAll(page, []byte("__VERSION__"), []byte(appManifest.Version)), nil
}

type enrollRequest struct {
	Name    string `json:"name"`
	Samples int    `json:"samples,omitempty"`
}

type removeRequest struct {
	Name string `json:"name"`
}

type modeRequest struct {
	Mode string `json:"mode"`
}

type cameraOption struct {
	DeviceID string `json:"device_id"`
	Label    string `json:"label"`
}

type cameraSelectRequest struct {
	DeviceID string `json:"device_id"`
}

type statusResponse struct {
	OK                  bool           `json:"ok"`
	ModelsReady         bool           `json:"models_ready"`
	ModelsError         string         `json:"models_error,omitempty"`
	Mode                string         `json:"mode"`
	CameraConnected     bool           `json:"camera_connected"`
	CameraAvailable     bool           `json:"camera_available"`
	CameraIndex         string         `json:"camera_index"`
	CameraDevices       []cameraOption `json:"camera_devices,omitempty"`
	CameraRetrySeconds  int            `json:"camera_retry_seconds"`
	CameraError         string         `json:"camera_error,omitempty"`
	People              int            `json:"people"`
	EnrollName          string         `json:"enroll_name,omitempty"`
	EnrollSamples       int            `json:"enroll_samples"`
	EnrollTargetSamples int            `json:"enroll_target_samples"`
	EnrollDurationSec   float32        `json:"enroll_duration_sec,omitempty"`
	EnrollElapsedSec    float32        `json:"enroll_elapsed_sec,omitempty"`
	// How the last recording ended. The number grows with each recording that ends.
	EnrollSeq           int     `json:"enroll_seq"`
	EnrollResult        string  `json:"enroll_result,omitempty"` // "saved", "no_face" or "failed"
	EnrollResultName    string  `json:"enroll_result_name,omitempty"`
	EnrollResultSamples int     `json:"enroll_result_samples,omitempty"`
	Faces               int     `json:"faces"`
	Hits                int     `json:"hits"`
	Best                float32 `json:"best"`
	InferMS             float64 `json:"infer_ms"`
	FPS                 float64 `json:"fps"`
	Timestamp           int64   `json:"ts"`
}

func (c *FaceClient) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	s := c.statusSnapshot()
	writeJSON(w, http.StatusOK, s)
}

func (c *FaceClient) handleResults(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	width, height := c.frameSize()
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":           true,
		"results":      c.resultsSnapshot(),
		"frame_width":  width,
		"frame_height": height,
		"mode":         string(c.currentMode()),
	})
}

func (c *FaceClient) handleMode(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req modeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	mode := PipelineMode(strings.ToLower(strings.TrimSpace(req.Mode)))
	if err := c.setMode(mode); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "mode": string(mode)})
}

func (c *FaceClient) handleReloadDB(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := c.reloadDB(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "people": len(c.store.List())})
}

func (c *FaceClient) handleEnrollStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req enrollRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	if err := c.startEnroll(req.Name, req.Samples); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	status := c.statusSnapshot()
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":                    true,
		"mode":                  status.Mode,
		"enroll_name":           status.EnrollName,
		"enroll_samples":        status.EnrollSamples,
		"enroll_target_samples": status.EnrollTargetSamples,
		"enroll_duration_sec":   status.EnrollDurationSec,
	})
}

func (c *FaceClient) handleEnrollCapture(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if err := c.captureEnrollSample(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	status := c.statusSnapshot()
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":                    true,
		"enroll_name":           status.EnrollName,
		"enroll_samples":        status.EnrollSamples,
		"enroll_target_samples": status.EnrollTargetSamples,
	})
}

func (c *FaceClient) handleEnrollCommit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req enrollRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err != io.EOF {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	name, samples, err := c.commitEnroll(req.Name)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":            true,
		"saved_name":    name,
		"saved_samples": samples,
		"people":        len(c.store.List()),
		"mode":          string(c.currentMode()),
	})
}

func (c *FaceClient) handleEnrollCancel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	c.cancelEnroll()
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "mode": string(c.currentMode())})
}

func (c *FaceClient) handlePeople(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	// Names and dates only: the face data itself stays in the app.
	type person struct {
		Name       string `json:"name"`
		Recordings int    `json:"recordings"` // 0: recorded by an earlier version, to be recorded again
		UpdatedAt  string `json:"updated_at,omitempty"`
	}
	stored := c.store.List()
	people := make([]person, 0, len(stored))
	for _, p := range stored {
		people = append(people, person{Name: p.Name, Recordings: len(p.Profiles), UpdatedAt: p.UpdatedAt})
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "people": people, "max_recordings": maxProfiles})
}

type settingsRequest struct {
	MatchThreshold float32 `json:"match_threshold"`
}

// handleSettings returns the match rules in force (GET) or changes the match threshold (POST).
func (c *FaceClient) handleSettings(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
	case http.MethodPost:
		var req settingsRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid json")
			return
		}
		if err := c.setMatchThreshold(req.MatchThreshold); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	rules := c.matchRules()
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":                  true,
		"match_threshold":     rules.Threshold,
		"single_threshold":    rules.SingleThreshold,
		"margin":              rules.Margin,
		"match_threshold_min": matchThresholdMin,
		"match_threshold_max": matchThresholdMax,
	})
}

func (c *FaceClient) handleRemove(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req removeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	removed, err := c.store.Remove(req.Name)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	_ = c.reloadDB()
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "removed": removed})
}

func (c *FaceClient) handleCameraSelect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req cameraSelectRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json")
		return
	}
	if err := c.selectCameraDevice(req.DeviceID); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	status := c.statusSnapshot()
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":           true,
		"camera_index": status.CameraIndex,
	})
}

func (c *FaceClient) handleVideoFeed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	c.addVideoFeed()
	defer c.removeVideoFeed()

	lastID := int64(-1)
	for {
		select {
		case <-r.Context().Done():
			return
		default:
		}

		jpg, id := c.waitForNextFrame(lastID, 2*time.Second)
		if len(jpg) == 0 || id == lastID {
			continue
		}
		lastID = id

		_, _ = w.Write([]byte("--frame\r\nContent-Type: image/jpeg\r\n\r\n"))
		_, _ = w.Write(jpg)
		_, _ = w.Write([]byte("\r\n"))
		flusher.Flush()
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"ok": false, "error": msg})
}
