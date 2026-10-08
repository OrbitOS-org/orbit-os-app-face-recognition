package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	orbitos "github.com/OrbitOS-org/orbit-os-sdk-go/v26/client"
	"github.com/OrbitOS-org/orbit-os-sdk-go/v26/logger"
)

const faceTag = "face_client"

type FaceClient struct {
	cfg      FaceClientConfig
	gravity  *orbitos.Client
	pipeline *FacePipeline
	store    *EmbeddingStore

	cameraDevice  string
	cameraClient  string
	lockedCam     *orbitos.LockedCamera
	streamCancel  context.CancelFunc
	cameraDevices []cameraOption
	// When the device last answered with its list of cameras (zero: never).
	cameraListedAt time.Time

	stateMu   sync.RWMutex
	frameCond *sync.Cond
	frameID   int64

	latestJPEG []byte
	lastStatus statusResponse

	lastResults         []FaceResult
	frameWidth          int // size of the frames the results refer to
	frameHeight         int
	lastTargetEmbedding []float32
	lastTargetDetection *FaceDetection
	lastProcessedAt     time.Time

	mode                PipelineMode
	enrollName          string
	enrollEmbeddings    [][]float32
	enrollTargetSamples int
	enrollStartTime     time.Time
	activeVideoFeeds    int

	// The page is registered with the Launcher (to remove it when the app stops).
	webRegistered bool

	// Lowest similarity accepted as a match; can be changed from the page.
	matchThreshold float32

	// How the last enrolment recording ended, for the page.
	enrollSeq           int
	enrollResult        string
	enrollResultName    string
	enrollResultSamples int

	// When true (Web UI on), TFLite models load on first web/video use, not at process start.
	lazyModelLoad bool
	modelsMu      sync.Mutex
	modelsReady   bool
	modelsLoadErr string
}

func NewFaceClient() *FaceClient {
	c := &FaceClient{
		cameraClient: "face-recognition-client",
	}
	c.frameCond = sync.NewCond(&c.stateMu)
	return c
}

func (c *FaceClient) Init() error {
	cfg, err := loadFaceClientConfig()
	if err != nil {
		return err
	}
	c.cfg = cfg
	c.mode = PipelineModeRecognize

	gravity, err := orbitos.NewClientAuto(cfg.GravityTCPHost)
	if err != nil {
		return fmt.Errorf("connect gravity: %w", err)
	}
	c.gravity = gravity

	c.store = NewEmbeddingStore(cfg.EmbeddingsFile)
	if err := c.store.Load(); err != nil {
		return err
	}
	c.loadSettings()

	c.pipeline = NewFacePipeline(cfg, c.gravity.AIManager)
	c.lazyModelLoad = webUIEnabled(&cfg)
	if !c.lazyModelLoad {
		if err := c.pipeline.EnsureModelsLoaded(); err != nil {
			return fmt.Errorf("load models: %w", err)
		}
		c.modelsMu.Lock()
		c.modelsReady = true
		c.modelsLoadErr = ""
		c.modelsMu.Unlock()
	}
	c.cameraDevice = strings.TrimSpace(cfg.CameraDeviceID)

	c.stateMu.Lock()
	c.lastStatus = statusResponse{
		OK:                  true,
		ModelsReady:         !c.lazyModelLoad,
		ModelsError:         "",
		Mode:                string(c.mode),
		CameraConnected:     false,
		CameraIndex:         c.cameraDevice,
		CameraAvailable:     false,
		CameraRetrySeconds:  c.cfg.CameraRetrySeconds,
		People:              len(c.store.List()),
		EnrollTargetSamples: c.cfg.EnrollTargetSamples,
		EnrollDurationSec:   c.cfg.EnrollDurationSeconds,
	}
	c.stateMu.Unlock()
	// The cameras are listed when the live view is first opened, not here:
	// listing can take a while and the page should open at once. Until then
	// the configured camera is the one on offer.
	c.useConfiguredCamera()

	if err := startWebUI(c); err != nil {
		return fmt.Errorf("web page: %w", err)
	}
	go c.streamFramesLoop()
	return nil
}

// Run waits until the app is asked to stop (Ctrl+C, or SIGTERM from Orbit OS),
// then takes its page off the Launcher and frees the camera.
func (c *FaceClient) Run() {
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop

	logger.Infof(faceTag, "stopping")
	c.cancelActiveStream()
	c.releaseCameraLock()
	if c.webRegistered {
		if err := c.gravity.AppHubManager.UnregisterService(); err != nil {
			logger.Warnf(faceTag, "UnregisterService: %v", err)
		}
	}
	c.gravity.Close()
}

// ensureModelsWarm loads models when modelsReady is false. The AI backend may unload models
// after idle time; inferModelGoneError + invalidateModelsState trigger a reload on the next frame.
func (c *FaceClient) ensureModelsWarm() error {
	c.modelsMu.Lock()
	defer c.modelsMu.Unlock()
	if c.modelsReady {
		return nil
	}
	err := c.pipeline.EnsureModelsLoaded()
	c.stateMu.Lock()
	if err != nil {
		c.modelsLoadErr = err.Error()
		c.modelsReady = false
		c.lastStatus.ModelsReady = false
		c.lastStatus.ModelsError = c.modelsLoadErr
	} else {
		c.modelsLoadErr = ""
		c.modelsReady = true
		c.lastStatus.ModelsReady = true
		c.lastStatus.ModelsError = ""
	}
	c.stateMu.Unlock()
	if err != nil {
		logger.Warnf(faceTag, "model load failed: %v", err)
	} else {
		logger.Infof(faceTag, "models ready")
	}
	return err
}

func inferModelGoneError(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "not loaded") ||
		strings.Contains(s, "no tensor schema") ||
		strings.Contains(s, "unknown model") ||
		(strings.Contains(s, "model") && strings.Contains(s, "unload"))
}

func (c *FaceClient) invalidateModelsState() {
	c.modelsMu.Lock()
	was := c.modelsReady
	c.modelsReady = false
	c.modelsLoadErr = ""
	c.modelsMu.Unlock()
	if was {
		logger.Infof(faceTag, "model cache invalidated (AI service likely evicted models); reloading")
	}
}

func (c *FaceClient) streamFramesLoop() {
	type queuedFrame struct {
		data []byte
	}

	for {
		if !c.hasActiveVideoFeeds() {
			c.setStreamPausedStatus()
			c.releaseCameraLock()
			time.Sleep(250 * time.Millisecond)
			continue
		}

		if c.cameraListStale() {
			c.updateCameraError("looking for cameras...")
			if _, err := c.refreshCameraInventory(); err != nil {
				c.updateCameraError(fmt.Sprintf("list cameras: %v", err))
				time.Sleep(time.Duration(c.cfg.CameraRetrySeconds) * time.Second)
				continue
			}
		}

		device, err := c.resolveCameraDevice()
		if err != nil {
			c.updateCameraError(err.Error())
			time.Sleep(time.Duration(c.cfg.CameraRetrySeconds) * time.Second)
			continue
		}
		if err := c.ensureCameraLock(device); err != nil {
			c.updateCameraError(fmt.Sprintf("lock camera %s: %v", device, err))
			time.Sleep(time.Duration(c.cfg.CameraRetrySeconds) * time.Second)
			continue
		}

		streamCtx, streamCancel := context.WithCancel(context.Background())
		c.setStreamCancel(streamCancel)
		c.stateMu.RLock()
		lockedCam := c.lockedCam
		c.stateMu.RUnlock()
		stream, err := lockedCam.StreamFrames(streamCtx, c.cfg.CameraFPS, c.cfg.CameraWidth, c.cfg.CameraHeight)
		if err != nil {
			c.updateCameraError(fmt.Sprintf("stream open: %v", err))
			logger.Warnf(faceTag, "open stream: %v", err)
			c.clearStreamCancel(streamCancel)
			time.Sleep(time.Duration(c.cfg.CameraRetrySeconds) * time.Second)
			continue
		}
		c.updateCameraConnected(true, "")

		// Decouple camera receive from inference; always process the newest frame.
		frameCh := make(chan queuedFrame, 1)
		errCh := make(chan error, 1)
		go func() {
			for {
				frame, recvErr := stream.Recv()
				if recvErr != nil {
					errCh <- recvErr
					return
				}
				data := frame.GetData()
				if len(data) == 0 {
					continue
				}
				copied := append([]byte(nil), data...)
				// The page always shows the camera frames as they arrive and
				// draws the boxes itself, from the results of the last frame
				// that was processed.
				c.stateMu.Lock()
				c.latestJPEG = append(c.latestJPEG[:0], copied...)
				c.frameID++
				c.frameCond.Broadcast()
				c.stateMu.Unlock()

				qf := queuedFrame{data: copied}
				select {
				case frameCh <- qf:
				default:
					select {
					case <-frameCh:
					default:
					}
					frameCh <- qf
				}
			}
		}()

		streamBroken := false
		for !streamBroken {
			if !c.hasActiveVideoFeeds() {
				streamBroken = true
				continue
			}
			select {
			case recvErr := <-errCh:
				if recvErr != io.EOF {
					c.updateCameraError(fmt.Sprintf("recv frame: %v", recvErr))
					logger.Warnf(faceTag, "recv frame: %v", recvErr)
				}
				streamBroken = true
			case qf := <-frameCh:
				if err := c.ensureModelsWarm(); err != nil {
					c.updateCameraError(fmt.Sprintf("models: %v", err))
					time.Sleep(500 * time.Millisecond)
					continue
				}
				start := time.Now()
				people := c.store.List()
				mode := c.currentMode()
				out, procErr := c.pipeline.ProcessFrame(qf.data, people, c.matchRules(), mode)
				inferMs := float64(time.Since(start).Milliseconds())
				if procErr != nil {
					if inferModelGoneError(procErr) {
						c.invalidateModelsState()
					}
					c.updateCameraError(fmt.Sprintf("process frame: %v", procErr))
					time.Sleep(200 * time.Millisecond)
					continue
				}

				hits, best := summarizeMatches(out.Results)
				now := time.Now()
				fps := 0.0
				shouldAutoCommit := false
				noFaceEnrolled := false
				c.stateMu.Lock()
				if !c.lastProcessedAt.IsZero() {
					dt := now.Sub(c.lastProcessedAt).Seconds()
					if dt > 0 {
						fps = 1.0 / dt
					}
				}
				c.lastProcessedAt = now
				c.lastResults = out.Results
				c.frameWidth, c.frameHeight = out.FrameWidth, out.FrameHeight
				c.lastTargetEmbedding = append(c.lastTargetEmbedding[:0], out.TargetEmbedding...)
				c.lastTargetDetection = out.TargetDetection

				// Auto-collect embeddings during the enrollment recording window.
				enrollElapsed := float32(0)
				if mode == PipelineModeEnroll && !c.enrollStartTime.IsZero() {
					dur := time.Duration(float64(time.Second) * float64(c.cfg.EnrollDurationSeconds))
					elapsed := now.Sub(c.enrollStartTime)
					enrollElapsed = float32(elapsed.Seconds())
					if elapsed <= dur {
						if len(out.TargetEmbedding) > 0 {
							c.enrollEmbeddings = append(c.enrollEmbeddings,
								append([]float32(nil), out.TargetEmbedding...))
						}
					} else if len(c.enrollEmbeddings) > 0 {
						// Window closed with samples — trigger auto-commit.
						c.enrollStartTime = time.Time{}
						shouldAutoCommit = true
					} else {
						// Window closed with no face seen: end the recording and say so.
						c.enrollStartTime = time.Time{}
						noFaceEnrolled = true
					}
				}

				cameraDevices := append([]cameraOption(nil), c.cameraDevices...)
				cameraAvailable := len(cameraDevices) > 0 || strings.TrimSpace(c.cameraDevice) != ""
				c.lastStatus = statusResponse{
					OK:                  true,
					Mode:                string(mode),
					CameraConnected:     true,
					CameraAvailable:     cameraAvailable,
					CameraIndex:         c.cameraDevice,
					CameraDevices:       cameraDevices,
					CameraRetrySeconds:  c.cfg.CameraRetrySeconds,
					CameraError:         "",
					People:              len(people),
					EnrollName:          c.enrollName,
					EnrollSamples:       len(c.enrollEmbeddings),
					EnrollTargetSamples: c.enrollTargetSamples,
					EnrollDurationSec:   c.cfg.EnrollDurationSeconds,
					EnrollElapsedSec:    enrollElapsed,
					Faces:               len(out.Results),
					Hits:                hits,
					Best:                best,
					InferMS:             inferMs,
					FPS:                 fps,
					Timestamp:           now.Unix(),
				}
				c.stateMu.Unlock()

				if shouldAutoCommit {
					if name, saved, err := c.commitEnroll(""); err != nil {
						logger.Warnf(faceTag, "auto-commit enroll failed: %v", err)
						c.cancelEnroll()
						c.setEnrollResult("failed", "", 0)
					} else {
						logger.Infof(faceTag, "auto-enrolled %q with %d samples", name, saved)
						c.setEnrollResult("saved", name, saved)
					}
				}
				if noFaceEnrolled {
					c.cancelEnroll()
					c.setEnrollResult("no_face", "", 0)
				}
			}
		}
		streamCancel()
		c.clearStreamCancel(streamCancel)
		c.releaseCameraLock()
		time.Sleep(200 * time.Millisecond)
	}
}

func (c *FaceClient) resolveCameraDevice() (string, error) {
	c.stateMu.RLock()
	selected := c.cameraDevice
	c.stateMu.RUnlock()
	if selected != "" {
		return selected, nil
	}
	devices, err := c.refreshCameraInventory()
	if err != nil {
		return "", fmt.Errorf("list cameras: %w", err)
	}
	if len(devices) == 0 {
		return "", fmt.Errorf("no camera devices connected")
	}
	c.stateMu.RLock()
	selected = c.cameraDevice
	c.stateMu.RUnlock()
	if selected == "" {
		return "", fmt.Errorf("no camera selected")
	}
	return selected, nil
}

func (c *FaceClient) currentMode() PipelineMode {
	c.stateMu.RLock()
	defer c.stateMu.RUnlock()
	return c.mode
}

func (c *FaceClient) setMode(mode PipelineMode) error {
	switch mode {
	case PipelineModeDetect, PipelineModeRecognize, PipelineModeEnroll:
	default:
		return fmt.Errorf("invalid mode %q", mode)
	}
	c.stateMu.Lock()
	c.mode = mode
	c.lastStatus.Mode = string(mode)
	c.stateMu.Unlock()
	return nil
}

func (c *FaceClient) statusSnapshot() statusResponse {
	c.stateMu.RLock()
	defer c.stateMu.RUnlock()
	s := c.lastStatus
	s.ModelsReady = c.modelsReady
	s.ModelsError = c.modelsLoadErr
	s.EnrollSeq = c.enrollSeq
	s.EnrollResult = c.enrollResult
	s.EnrollResultName = c.enrollResultName
	s.EnrollResultSamples = c.enrollResultSamples
	if s.CameraConnected && c.lastProcessedAt.IsZero() {
		s.CameraConnected = false
	}
	return s
}

func (c *FaceClient) resultsSnapshot() []FaceResult {
	c.stateMu.RLock()
	defer c.stateMu.RUnlock()
	out := make([]FaceResult, len(c.lastResults))
	copy(out, c.lastResults)
	return out
}

// frameSize returns the size of the frames the results refer to.
func (c *FaceClient) frameSize() (width, height int) {
	c.stateMu.RLock()
	defer c.stateMu.RUnlock()
	return c.frameWidth, c.frameHeight
}

func (c *FaceClient) waitForNextFrame(lastFrameID int64, timeout time.Duration) ([]byte, int64) {
	deadline := time.Now().Add(timeout)
	for {
		c.stateMu.RLock()
		currentID := c.frameID
		if currentID > lastFrameID && len(c.latestJPEG) > 0 {
			frame := append([]byte(nil), c.latestJPEG...)
			c.stateMu.RUnlock()
			return frame, currentID
		}
		c.stateMu.RUnlock()

		if timeout > 0 && time.Now().After(deadline) {
			c.stateMu.RLock()
			id := c.frameID
			c.stateMu.RUnlock()
			return nil, id
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (c *FaceClient) reloadDB() error {
	if err := c.store.Load(); err != nil {
		return err
	}
	c.stateMu.Lock()
	c.lastStatus.People = len(c.store.List())
	c.stateMu.Unlock()
	return nil
}

func (c *FaceClient) startEnroll(name string, samples int) error {
	name = strings.TrimSpace(name)
	// Estimate target sample count from duration × fps; allow explicit override.
	targetSamples := int(c.cfg.EnrollDurationSeconds * float32(c.cfg.CameraFPS))
	if targetSamples < 1 {
		targetSamples = c.cfg.EnrollTargetSamples
	}
	if samples > 0 {
		targetSamples = samples
	}
	c.stateMu.Lock()
	c.mode = PipelineModeEnroll
	c.enrollName = name
	c.enrollEmbeddings = c.enrollEmbeddings[:0]
	c.enrollTargetSamples = targetSamples
	c.enrollStartTime = time.Now()
	c.lastStatus.Mode = string(c.mode)
	c.lastStatus.EnrollName = name
	c.lastStatus.EnrollSamples = 0
	c.lastStatus.EnrollTargetSamples = targetSamples
	c.lastStatus.EnrollDurationSec = c.cfg.EnrollDurationSeconds
	c.lastStatus.EnrollElapsedSec = 0
	c.stateMu.Unlock()
	return nil
}

func (c *FaceClient) captureEnrollSample() error {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	if c.mode != PipelineModeEnroll {
		return fmt.Errorf("not in enroll mode")
	}
	if len(c.lastTargetEmbedding) == 0 {
		return fmt.Errorf("no face available to capture")
	}
	emb := append([]float32(nil), c.lastTargetEmbedding...)
	c.enrollEmbeddings = append(c.enrollEmbeddings, emb)
	c.lastStatus.EnrollSamples = len(c.enrollEmbeddings)
	return nil
}

func (c *FaceClient) commitEnroll(nameOverride string) (string, int, error) {
	c.stateMu.Lock()
	name := c.enrollName
	samples := make([][]float32, len(c.enrollEmbeddings))
	for i := range c.enrollEmbeddings {
		samples[i] = append([]float32(nil), c.enrollEmbeddings[i]...)
	}
	c.stateMu.Unlock()

	nameOverride = strings.TrimSpace(nameOverride)
	if nameOverride != "" {
		name = nameOverride
	}
	if len(samples) == 0 {
		return "", 0, fmt.Errorf("no enroll samples to commit")
	}
	if name == "" {
		return "", 0, fmt.Errorf("name is required")
	}
	avg := averageEmbeddingsBatch(samples)
	if len(avg) == 0 {
		return "", 0, fmt.Errorf("invalid enroll samples")
	}
	if err := c.store.AddProfile(name, avg, len(samples)); err != nil {
		return "", 0, err
	}

	c.stateMu.Lock()
	savedSamples := len(c.enrollEmbeddings)
	c.enrollEmbeddings = c.enrollEmbeddings[:0]
	c.enrollName = ""
	c.enrollStartTime = time.Time{}
	c.mode = PipelineModeRecognize
	c.lastStatus.Mode = string(c.mode)
	c.lastStatus.People = len(c.store.List())
	c.lastStatus.EnrollName = ""
	c.lastStatus.EnrollSamples = 0
	c.lastStatus.EnrollTargetSamples = c.cfg.EnrollTargetSamples
	c.lastStatus.EnrollElapsedSec = 0
	c.stateMu.Unlock()
	return name, savedSamples, nil
}

// setEnrollResult records how a recording ended, for the page to show.
func (c *FaceClient) setEnrollResult(result, name string, samples int) {
	c.stateMu.Lock()
	c.enrollSeq++
	c.enrollResult = result
	c.enrollResultName = name
	c.enrollResultSamples = samples
	c.stateMu.Unlock()
}

func (c *FaceClient) cancelEnroll() {
	c.stateMu.Lock()
	c.enrollEmbeddings = c.enrollEmbeddings[:0]
	c.enrollName = ""
	c.enrollStartTime = time.Time{}
	c.mode = PipelineModeRecognize
	c.lastStatus.Mode = string(c.mode)
	c.lastStatus.EnrollName = ""
	c.lastStatus.EnrollSamples = 0
	c.lastStatus.EnrollTargetSamples = c.cfg.EnrollTargetSamples
	c.lastStatus.EnrollElapsedSec = 0
	c.stateMu.Unlock()
}

func (c *FaceClient) updateCameraError(errText string) {
	c.stateMu.Lock()
	c.lastStatus.CameraConnected = false
	c.lastStatus.CameraError = errText
	c.lastStatus.Mode = string(c.mode)
	c.lastStatus.CameraRetrySeconds = c.cfg.CameraRetrySeconds
	c.lastStatus.CameraIndex = c.cameraDevice
	c.lastStatus.CameraDevices = append([]cameraOption(nil), c.cameraDevices...)
	c.lastStatus.CameraAvailable = len(c.cameraDevices) > 0
	c.stateMu.Unlock()
}

func (c *FaceClient) updateCameraConnected(connected bool, errText string) {
	c.stateMu.Lock()
	c.lastStatus.CameraConnected = connected
	c.lastStatus.CameraError = errText
	c.lastStatus.Mode = string(c.mode)
	c.lastStatus.CameraRetrySeconds = c.cfg.CameraRetrySeconds
	c.lastStatus.CameraIndex = c.cameraDevice
	c.lastStatus.CameraDevices = append([]cameraOption(nil), c.cameraDevices...)
	c.lastStatus.CameraAvailable = len(c.cameraDevices) > 0
	c.stateMu.Unlock()
}

// Listing the cameras makes the device look at every video node it has, which
// on some boards takes well over ten seconds.
const (
	cameraListTimeout = 45 * time.Second
	cameraListMaxAge  = 2 * time.Minute
)

// cameraListStale reports whether the cameras should be listed again.
func (c *FaceClient) cameraListStale() bool {
	c.stateMu.RLock()
	defer c.stateMu.RUnlock()
	return c.cameraListedAt.IsZero() || time.Since(c.cameraListedAt) > cameraListMaxAge
}

// useConfiguredCamera offers the configured camera when no list is known.
// It returns the cameras on offer afterwards.
func (c *FaceClient) useConfiguredCamera() []cameraOption {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	if len(c.cameraDevices) == 0 {
		if device := strings.TrimSpace(c.cfg.CameraDeviceID); device != "" {
			c.cameraDevices = []cameraOption{{DeviceID: device, Label: device}}
			if c.cameraDevice == "" {
				c.cameraDevice = device
			}
		}
	}
	c.lastStatus.CameraAvailable = len(c.cameraDevices) > 0
	c.lastStatus.CameraDevices = append([]cameraOption(nil), c.cameraDevices...)
	c.lastStatus.CameraIndex = c.cameraDevice
	return append([]cameraOption(nil), c.cameraDevices...)
}

func (c *FaceClient) refreshCameraInventory() ([]cameraOption, error) {
	ctx, cancel := context.WithTimeout(context.Background(), cameraListTimeout)
	defer cancel()

	devices, err := c.gravity.CameraManager.ListDevices(ctx)
	if err != nil {
		// Carry on with the cameras already known, or with the configured
		// one, and try the list again on the next start of the live view.
		if known := c.useConfiguredCamera(); len(known) > 0 {
			logger.Warnf(faceTag, "list cameras: %v; using %d known camera(s)", err, len(known))
			return known, nil
		}
		return nil, err
	}
	options := make([]cameraOption, 0, len(devices))
	for _, d := range devices {
		deviceID := strings.TrimSpace(d.DeviceID)
		if deviceID == "" {
			continue
		}
		label := deviceID
		if card := strings.TrimSpace(d.Card); card != "" {
			label = fmt.Sprintf("%s (%s)", deviceID, card)
		}
		options = append(options, cameraOption{DeviceID: deviceID, Label: label})
	}

	c.stateMu.Lock()
	c.cameraListedAt = time.Now()
	c.cameraDevices = options
	if len(options) == 0 {
		c.cameraDevice = ""
		c.lockedCam = nil
	} else if c.cameraDevice == "" || !containsCamera(options, c.cameraDevice) {
		c.cameraDevice = options[0].DeviceID
		c.lockedCam = nil
	}
	c.lastStatus.CameraAvailable = len(options) > 0
	c.lastStatus.CameraDevices = append([]cameraOption(nil), options...)
	c.lastStatus.CameraIndex = c.cameraDevice
	c.stateMu.Unlock()
	return options, nil
}

func (c *FaceClient) cameraStateSnapshot() (string, []cameraOption, bool) {
	c.stateMu.RLock()
	defer c.stateMu.RUnlock()
	return c.cameraDevice, append([]cameraOption(nil), c.cameraDevices...), c.lockedCam != nil
}

func (c *FaceClient) selectCameraDevice(deviceID string) error {
	deviceID = strings.TrimSpace(deviceID)
	if deviceID == "" {
		return fmt.Errorf("device_id is required")
	}
	// The cameras on offer in the page come from the last list; ask the
	// device again only when that list is old.
	_, options, _ := c.cameraStateSnapshot()
	if len(options) == 0 || c.cameraListStale() {
		var err error
		if options, err = c.refreshCameraInventory(); err != nil {
			return fmt.Errorf("list cameras: %w", err)
		}
	}
	if !containsCamera(options, deviceID) {
		return fmt.Errorf("camera %q not found", deviceID)
	}

	var oldCam *orbitos.LockedCamera
	c.stateMu.Lock()
	if c.lockedCam != nil && c.cameraDevice != "" && c.cameraDevice != deviceID {
		oldCam = c.lockedCam
	}
	c.cameraDevice = deviceID
	c.lockedCam = nil
	c.lastStatus.CameraConnected = false
	c.lastStatus.CameraError = "switching camera"
	c.lastStatus.CameraIndex = c.cameraDevice
	c.stateMu.Unlock()

	if oldCam != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		_ = oldCam.Unlock(ctx)
		cancel()
	}
	c.cancelActiveStream()
	return nil
}

func (c *FaceClient) ensureCameraLock(device string) error {
	c.stateMu.RLock()
	locked := c.lockedCam != nil && c.cameraDevice == device
	c.stateMu.RUnlock()
	if locked {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	lockedCam, err := c.gravity.CameraManager.Lock(ctx, device, c.cameraClient)
	if err != nil {
		return err
	}

	c.stateMu.Lock()
	c.lockedCam = lockedCam
	c.lastStatus.CameraIndex = c.cameraDevice
	c.stateMu.Unlock()
	return nil
}

func (c *FaceClient) setStreamCancel(cancel context.CancelFunc) {
	c.stateMu.Lock()
	c.streamCancel = cancel
	c.stateMu.Unlock()
}

func (c *FaceClient) clearStreamCancel(cancel context.CancelFunc) {
	c.stateMu.Lock()
	if fmt.Sprintf("%p", c.streamCancel) == fmt.Sprintf("%p", cancel) {
		c.streamCancel = nil
	}
	c.stateMu.Unlock()
}

func (c *FaceClient) cancelActiveStream() {
	c.stateMu.RLock()
	cancel := c.streamCancel
	c.stateMu.RUnlock()
	if cancel != nil {
		cancel()
	}
}

func (c *FaceClient) hasActiveVideoFeeds() bool {
	c.stateMu.RLock()
	defer c.stateMu.RUnlock()
	return c.activeVideoFeeds > 0
}

func (c *FaceClient) addVideoFeed() {
	c.stateMu.Lock()
	c.activeVideoFeeds++
	c.stateMu.Unlock()
}

func (c *FaceClient) removeVideoFeed() {
	c.stateMu.Lock()
	if c.activeVideoFeeds > 0 {
		c.activeVideoFeeds--
	}
	c.stateMu.Unlock()
}

func (c *FaceClient) releaseCameraLock() {
	c.stateMu.RLock()
	lockedCam := c.lockedCam
	c.stateMu.RUnlock()
	if lockedCam == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	_ = lockedCam.Unlock(ctx)
	cancel()
	c.stateMu.Lock()
	c.lockedCam = nil
	c.stateMu.Unlock()
}

func (c *FaceClient) setStreamPausedStatus() {
	c.stateMu.Lock()
	defer c.stateMu.Unlock()
	c.lastStatus.CameraConnected = false
	if c.lastStatus.CameraError == "" {
		c.lastStatus.CameraError = "stream paused"
	}
	c.lastStatus.CameraIndex = c.cameraDevice
	c.lastStatus.CameraDevices = append([]cameraOption(nil), c.cameraDevices...)
	c.lastStatus.CameraAvailable = len(c.cameraDevices) > 0
}

func containsCamera(options []cameraOption, deviceID string) bool {
	for _, opt := range options {
		if opt.DeviceID == deviceID {
			return true
		}
	}
	return false
}

func summarizeMatches(results []FaceResult) (hits int, best float32) {
	best = 0
	for _, r := range results {
		if r.Match != nil && r.Match.Accepted {
			hits++
			if r.Match.Similarity > best {
				best = r.Match.Similarity
			}
		}
	}
	return hits, best
}

func averageEmbeddingsBatch(samples [][]float32) []float32 {
	if len(samples) == 0 {
		return nil
	}
	n := len(samples[0])
	if n == 0 {
		return nil
	}
	out := make([]float32, n)
	valid := 0
	for _, sample := range samples {
		if len(sample) < n {
			continue
		}
		valid++
		for i := 0; i < n; i++ {
			out[i] += sample[i]
		}
	}
	if valid == 0 {
		return nil
	}
	scale := float32(1.0 / float32(valid))
	for i := range out {
		out[i] *= scale
	}
	return normalizeL2(out)
}
