package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"

	aiv26 "github.com/OrbitOS-org/orbit-os-sdk-go/v26/api/ai_service/v26"
	orbitos "github.com/OrbitOS-org/orbit-os-sdk-go/v26/client"
	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/math/f64"
)

// The pipeline turns a camera frame into faces, and faces into names:
//
//	detect  BlazeFace finds the faces and, for each, the eyes, the nose and the mouth
//	follow  each face keeps its number from one frame to the next
//	align   the face is rotated and scaled to the position the embedding model expects
//	embed   SFace turns the aligned face into a list of numbers
//	match   that list is compared with the enrolled people (recognition.go)
//
// Detection runs on every frame. Embedding, the costly part, runs only when it
// is needed: for a new face, again from time to time, and while enrolling.

// FaceDetection is one face found in a frame, in frame pixels.
type FaceDetection struct {
	XMin  int32   `json:"xmin"`
	YMin  int32   `json:"ymin"`
	XMax  int32   `json:"xmax"`
	YMax  int32   `json:"ymax"`
	Score float32 `json:"score"`

	// The eye on the left of the picture, the eye on the right, the nose tip
	// and the centre of the mouth, as given by the detector.
	keypoints    [alignPoints][2]float32
	hasKeypoints bool
}

// FaceResult is what the page receives for each face.
type FaceResult struct {
	ID        int           `json:"id"` // the same number while the face stays in view
	Detection FaceDetection `json:"detection"`
	Match     *MatchResult  `json:"match,omitempty"`
	Target    bool          `json:"target,omitempty"` // the face being enrolled
}

type PipelineMode string

const (
	PipelineModeDetect    PipelineMode = "detect"
	PipelineModeRecognize PipelineMode = "recognize"
	PipelineModeEnroll    PipelineMode = "enroll"
)

type PipelineOutput struct {
	Results         []FaceResult
	TargetEmbedding []float32
	TargetDetection *FaceDetection
	FrameWidth      int
	FrameHeight     int
}

const (
	// A face is identified again after this long: sooner while it is not
	// recognised, so that a name appears quickly once the face turns to the camera.
	recheckKnown   = time.Second
	recheckUnknown = 300 * time.Millisecond

	trackMinOverlap = 0.3         // overlap with its last position for a face to keep its number
	trackKeep       = time.Second // a face not seen for this long is forgotten
	nmsOverlap      = 0.3         // detections overlapping more than this are the same face
	minFacePixels   = 20          // smaller boxes are not faces worth handling
	inferTimeout    = 10 * time.Second
)

type faceTrack struct {
	id         int
	box        FaceDetection
	seenAt     time.Time
	match      *MatchResult
	identified bool
	matchedAt  time.Time
}

type FacePipeline struct {
	cfg         FaceClientConfig
	ai          *orbitos.AIManager
	detectModel *orbitos.AIModel
	embedModel  *orbitos.AIModel
	detectInput []int32
	embedInput  []int32
	detectDtype aiv26.TensorDataType
	embedDtype  aiv26.TensorDataType

	// Reused from frame to frame.
	detectCanvas *image.RGBA
	tracks       []*faceTrack
	lastTrackID  int
}

func NewFacePipeline(cfg FaceClientConfig, ai *orbitos.AIManager) *FacePipeline {
	return &FacePipeline{cfg: cfg, ai: ai}
}

func (p *FacePipeline) EnsureModelsLoaded() error {
	detectModel, err := p.loadModel(p.cfg.DetectModelID, p.cfg.DetectModelPath)
	if err != nil {
		return err
	}
	embedModel, err := p.loadModel(p.cfg.EmbedModelID, p.cfg.EmbedModelPath)
	if err != nil {
		return err
	}
	p.detectModel = detectModel
	p.embedModel = embedModel
	p.detectInput, p.detectDtype = resolveInputTensor(detectModel.Response.GetInputs(), []int32{1, 192, 192, 3})
	p.embedInput, p.embedDtype = resolveInputTensor(embedModel.Response.GetInputs(), []int32{1, 3, 112, 112})
	return nil
}

func (p *FacePipeline) loadModel(modelID, modelPath string) (*orbitos.AIModel, error) {
	mode := aiv26.ExecutionMode_EXEC_CPU
	switch p.cfg.ExecutionMode {
	case "gpu":
		mode = aiv26.ExecutionMode_EXEC_GPU
	case "high_threads":
		mode = aiv26.ExecutionMode_EXEC_HIGH_THREADS
	}

	// The detector is a TensorFlow Lite model, the embedding model an ONNX one; Orbit OS runs both.
	backend := aiv26.ModelBackend_TFLITE
	if strings.EqualFold(filepath.Ext(modelPath), ".onnx") {
		backend = aiv26.ModelBackend_ONNX
	}
	if p.cfg.UploadModels && fileExists(modelPath) {
		return p.ai.UploadAndLoadModel(modelID, modelPath, backend, mode)
	}
	return p.ai.LoadModel(modelID, modelPath, backend, mode)
}

// ProcessFrame finds the faces in one camera frame and, depending on the mode,
// identifies them or prepares the face being enrolled.
func (p *FacePipeline) ProcessFrame(jpegFrame []byte, people []personEmbedding, rules matchRules, mode PipelineMode) (*PipelineOutput, error) {
	frame, err := decodeFrame(jpegFrame)
	if err != nil {
		return nil, err
	}
	detections, err := p.detectFaces(frame)
	if err != nil {
		return nil, err
	}

	now := time.Now()
	tracks := p.follow(detections, now)
	results := make([]FaceResult, len(detections))
	for i, d := range detections {
		results[i] = FaceResult{ID: tracks[i].id, Detection: d}
	}
	out := &PipelineOutput{
		Results:     results,
		FrameWidth:  frame.Bounds().Dx(),
		FrameHeight: frame.Bounds().Dy(),
	}

	switch mode {
	case PipelineModeEnroll:
		// Only the largest face is enrolled: the person in front of the camera.
		target := largestFace(detections)
		if target < 0 {
			break
		}
		emb, err := p.faceEmbedding(frame, detections[target])
		if err != nil {
			return nil, err
		}
		results[target].Target = true
		det := detections[target]
		out.TargetEmbedding = emb
		out.TargetDetection = &det

	case PipelineModeRecognize:
		for i, t := range tracks {
			if !t.identified || now.Sub(t.matchedAt) >= t.recheckAfter() {
				emb, err := p.faceEmbedding(frame, detections[i])
				if err != nil {
					return nil, err
				}
				t.match = findBestMatch(emb, people, rules)
				t.identified = true
				t.matchedAt = now
			}
			results[i].Match = t.match
		}
	}
	return out, nil
}

func (t *faceTrack) recheckAfter() time.Duration {
	if t.match != nil && t.match.Accepted {
		return recheckKnown
	}
	return recheckUnknown
}

func largestFace(dets []FaceDetection) int {
	best, bestArea := -1, int32(0)
	for i, d := range dets {
		if area := (d.XMax - d.XMin) * (d.YMax - d.YMin); area > bestArea {
			best, bestArea = i, area
		}
	}
	return best
}

// follow gives each detection the track of the face it continues, or a new
// one. The result has one track for each detection, in the same order.
func (p *FacePipeline) follow(dets []FaceDetection, now time.Time) []*faceTrack {
	// A face not seen for a while is forgotten; if it comes back it is a new face.
	kept := p.tracks[:0]
	for _, t := range p.tracks {
		if now.Sub(t.seenAt) <= trackKeep {
			kept = append(kept, t)
		}
	}
	p.tracks = kept

	assigned := make([]*faceTrack, len(dets))
	taken := make(map[*faceTrack]bool, len(p.tracks))
	for i, d := range dets {
		var best *faceTrack
		bestOverlap := float32(trackMinOverlap)
		for _, t := range p.tracks {
			if taken[t] {
				continue
			}
			if overlap := iou(t.box, d); overlap >= bestOverlap {
				best, bestOverlap = t, overlap
			}
		}
		if best == nil {
			p.lastTrackID++
			best = &faceTrack{id: p.lastTrackID}
			p.tracks = append(p.tracks, best)
		}
		taken[best] = true
		best.box = d
		best.seenAt = now
		assigned[i] = best
	}
	return assigned
}

// ---------------------------------------------------------------- detection

// letterbox says how a frame was placed inside the square the detector takes:
// scaled by the same factor in both directions and centred, with black bars
// filling the rest. The model was trained on pictures prepared this way.
type letterbox struct {
	scale      float64 // model pixels for one frame pixel
	padX, padY float64 // width of the bars, in model pixels
}

func (l letterbox) toFrame(x, y float32) (float32, float32) {
	return float32((float64(x) - l.padX) / l.scale), float32((float64(y) - l.padY) / l.scale)
}

func fitLetterbox(frameW, frameH, netW, netH int) (letterbox, image.Rectangle) {
	scale := math.Min(float64(netW)/float64(frameW), float64(netH)/float64(frameH))
	w := int(math.Round(float64(frameW) * scale))
	h := int(math.Round(float64(frameH) * scale))
	x0 := (netW - w) / 2
	y0 := (netH - h) / 2
	return letterbox{scale: scale, padX: float64(x0), padY: float64(y0)}, image.Rect(x0, y0, x0+w, y0+h)
}

func (p *FacePipeline) detectFaces(frame *image.RGBA) ([]FaceDetection, error) {
	netW, netH, channels := tensorDimensions(p.detectInput)
	if channels != 3 || netW <= 0 || netH <= 0 {
		return nil, fmt.Errorf("unsupported detector input %v", p.detectInput)
	}
	if p.detectCanvas == nil || p.detectCanvas.Bounds().Dx() != netW || p.detectCanvas.Bounds().Dy() != netH {
		p.detectCanvas = image.NewRGBA(image.Rect(0, 0, netW, netH))
	}
	canvas := p.detectCanvas
	for i := range canvas.Pix {
		canvas.Pix[i] = 0
	}
	box, inside := fitLetterbox(frame.Bounds().Dx(), frame.Bounds().Dy(), netW, netH)
	scaleInto(canvas, inside, frame)

	// BlazeFace takes each colour value between -1 and 1.
	input := pixelsToTensor(canvas, false, func(v uint8) float32 { return float32(v)/127.5 - 1 })

	ctx, cancel := context.WithTimeout(context.Background(), inferTimeout)
	defer cancel()
	resp, err := p.detectModel.Infer(ctx, input, p.detectInput, p.detectDtype)
	if err != nil {
		return nil, err
	}
	tensors := collectOutputTensors(bytesToFloat32(resp.GetOutputData()), resp.GetOutputShape(), resp.GetNamedOutputs())
	return decodeBlazeFace(tensors, box, frame.Bounds().Dx(), frame.Bounds().Dy(), netW, netH, p.cfg.ScoreThreshold, p.cfg.MaxFaces), nil
}

// scaleInto draws the whole frame, scaled, into one rectangle of dst.
//
// A camera frame is several times larger than what the detector takes, so
// each pixel written is the average of the block of frame pixels it covers:
// every frame pixel is read once, and fine detail does not turn into noise.
func scaleInto(dst *image.RGBA, where image.Rectangle, frame *image.RGBA) {
	fw, fh := frame.Bounds().Dx(), frame.Bounds().Dy()
	w, h := where.Dx(), where.Dy()
	if w <= 0 || h <= 0 || fw < w || fh < h {
		xdraw.ApproxBiLinear.Scale(dst, where, frame, frame.Bounds(), xdraw.Src, nil)
		return
	}
	for y := 0; y < h; y++ {
		y0, y1 := y*fh/h, (y+1)*fh/h
		out := dst.Pix[dst.PixOffset(where.Min.X, where.Min.Y+y):]
		for x := 0; x < w; x++ {
			x0, x1 := x*fw/w, (x+1)*fw/w
			var r, g, b uint32
			for sy := y0; sy < y1; sy++ {
				row := frame.Pix[sy*frame.Stride+x0*4 : sy*frame.Stride+x1*4]
				for i := 0; i < len(row); i += 4 {
					r += uint32(row[i])
					g += uint32(row[i+1])
					b += uint32(row[i+2])
				}
			}
			n := uint32((x1 - x0) * (y1 - y0))
			out[x*4], out[x*4+1], out[x*4+2], out[x*4+3] = uint8(r/n), uint8(g/n), uint8(b/n), 255
		}
	}
}

// pixelsToTensor writes the picture as float32 values, ready to send to a
// model: pixel after pixel (R, G, B of each), or, with planes set, the whole
// red plane, then the green, then the blue.
func pixelsToTensor(img *image.RGBA, planes bool, value func(uint8) float32) []byte {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	var table [256]uint32
	for v := range table {
		table[v] = math.Float32bits(value(uint8(v)))
	}
	out := make([]byte, w*h*3*4)
	next, plane := 12, 4 // bytes from one pixel to the next, and from one colour to the next
	if planes {
		next, plane = 4, w*h*4
	}
	at := 0
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride : y*img.Stride+w*4]
		for x := 0; x < w; x++ {
			binary.LittleEndian.PutUint32(out[at:], table[row[x*4]])
			binary.LittleEndian.PutUint32(out[at+plane:], table[row[x*4+1]])
			binary.LittleEndian.PutUint32(out[at+2*plane:], table[row[x*4+2]])
			at += next
		}
	}
	return out
}

type floatTensor struct {
	Data  []float32
	Rows  int
	Cols  int
	Valid bool
}

func collectOutputTensors(primary []float32, primaryShape []int32, named map[string][]byte) []floatTensor {
	out := make([]floatTensor, 0, 1+len(named))
	if t := tensorFromShape(primary, primaryShape); t.Valid {
		out = append(out, t)
	}
	for _, raw := range named {
		values := bytesToFloat32(raw)
		if len(values) == 0 {
			continue
		}
		// The box tensor is large (16 values for each anchor); the score
		// tensor has one value for each anchor and must not be read as boxes.
		if len(values) >= 4096 && len(values)%16 == 0 {
			out = append(out, floatTensor{Data: values, Rows: len(values) / 16, Cols: 16, Valid: true})
			continue
		}
		out = append(out, floatTensor{Data: values, Rows: len(values), Cols: 1, Valid: true})
	}
	return out
}

func tensorFromShape(data []float32, shape []int32) floatTensor {
	if len(data) == 0 || len(shape) < 2 || len(shape) > 4 {
		return floatTensor{}
	}
	r, c := int(shape[0]), int(shape[1])
	if len(shape) > 2 {
		r, c = int(shape[1]), int(shape[2])
	}
	if r > 0 && c > 0 && r*c == len(data) {
		return floatTensor{Data: data, Rows: r, Cols: c, Valid: true}
	}
	return floatTensor{}
}

// decodeBlazeFace turns the raw output of the detector (16 numbers and one
// score for each anchor) into faces in frame pixels.
func decodeBlazeFace(tensors []floatTensor, box letterbox, frameW, frameH, netW, netH int, scoreThreshold float32, maxFaces int) []FaceDetection {
	var reg, cls *floatTensor
	for i := range tensors {
		t := &tensors[i]
		if !t.Valid {
			continue
		}
		if t.Cols == 16 && t.Rows > 16 {
			reg = t
		} else if t.Cols == 1 && t.Rows > 16 {
			cls = t
		}
	}
	if reg == nil || cls == nil || reg.Rows != cls.Rows {
		return nil
	}
	anchors := buildBlazeAnchors(reg.Rows, netW, netH)
	if len(anchors) != reg.Rows {
		return nil
	}

	var candidates []FaceDetection
	for i := 0; i < reg.Rows; i++ {
		score := sigmoid(cls.Data[i])
		if score < scoreThreshold {
			continue
		}
		// Every value is an offset from the centre of the anchor, in model pixels.
		anchorX, anchorY := anchors[i][0]*float32(netW), anchors[i][1]*float32(netH)
		raw := reg.Data[i*reg.Cols : (i+1)*reg.Cols]
		cx, cy, w, h := raw[0]+anchorX, raw[1]+anchorY, raw[2], raw[3]

		x1, y1 := box.toFrame(cx-w/2, cy-h/2)
		x2, y2 := box.toFrame(cx+w/2, cy+h/2)
		d := FaceDetection{
			XMin:  int32(math.Round(clamp(float64(x1), 0, float64(frameW-1)))),
			YMin:  int32(math.Round(clamp(float64(y1), 0, float64(frameH-1)))),
			XMax:  int32(math.Round(clamp(float64(x2), 0, float64(frameW-1)))),
			YMax:  int32(math.Round(clamp(float64(y2), 0, float64(frameH-1)))),
			Score: score,
		}
		if d.XMax-d.XMin < minFacePixels || d.YMax-d.YMin < minFacePixels {
			continue
		}
		// After the box come the keypoints: the two eyes, the nose tip, the mouth, the two ears.
		for k := 0; k < alignPoints; k++ {
			x, y := box.toFrame(raw[4+2*k]+anchorX, raw[5+2*k]+anchorY)
			d.keypoints[k] = [2]float32{x, y}
		}
		d.hasKeypoints = true
		candidates = append(candidates, d)
	}

	keep := nms(candidates, nmsOverlap)
	dets := make([]FaceDetection, 0, len(keep))
	for _, idx := range keep {
		dets = append(dets, candidates[idx])
	}
	if maxFaces > 0 && len(dets) > maxFaces {
		dets = dets[:maxFaces]
	}
	return dets
}

// sigmoid turns the raw score of the detector into a probability.
func sigmoid(v float32) float32 {
	return float32(1 / (1 + math.Exp(-clamp(float64(v), -100, 100))))
}

// buildBlazeAnchors returns the centre of every anchor, as a fraction of the
// input size, for the two BlazeFace models: full range (one anchor for every
// 4 pixels) and short range (128x128, 896 anchors).
func buildBlazeAnchors(numAnchors, netW, netH int) [][2]float32 {
	anchors := make([][2]float32, 0, numAnchors)
	addGrid := func(stride, perCell int) {
		for y := 0; y < netH/stride; y++ {
			ay := (float32(y) + 0.5) * float32(stride) / float32(netH)
			for x := 0; x < netW/stride; x++ {
				ax := (float32(x) + 0.5) * float32(stride) / float32(netW)
				for i := 0; i < perCell; i++ {
					anchors = append(anchors, [2]float32{ax, ay})
				}
			}
		}
	}
	if numAnchors == (netW/4)*(netH/4) {
		addGrid(4, 1)
	} else if numAnchors == 896 && netW == 128 && netH == 128 {
		addGrid(8, 2)
		addGrid(16, 6)
	}
	if len(anchors) != numAnchors {
		return nil
	}
	return anchors
}

// nms keeps the best of the detections that cover the same face. It returns
// their positions in dets, best first.
func nms(dets []FaceDetection, overlap float32) []int {
	order := make([]int, len(dets))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return dets[order[i]].Score > dets[order[j]].Score })

	var keep []int
	for len(order) > 0 {
		best := order[0]
		keep = append(keep, best)
		rest := order[1:]
		order = order[:0:0]
		for _, j := range rest {
			if iou(dets[best], dets[j]) < overlap {
				order = append(order, j)
			}
		}
	}
	return keep
}

func iou(a, b FaceDetection) float32 {
	w := float32(min(a.XMax, b.XMax) - max(a.XMin, b.XMin))
	h := float32(min(a.YMax, b.YMax) - max(a.YMin, b.YMin))
	if w <= 0 || h <= 0 {
		return 0
	}
	inter := w * h
	areaA := float32((a.XMax - a.XMin) * (a.YMax - a.YMin))
	areaB := float32((b.XMax - b.XMin) * (b.YMax - b.YMin))
	return inter / (areaA + areaB - inter + 1e-6)
}

// ---------------------------------------------------------------- embedding

// Where the eyes, the nose tip and the centre of the mouth are expected in the
// 112x112 picture given to the embedding model (the ArcFace reference
// positions; the mouth is the middle of its two corners).
const alignPoints = 4

var alignTemplate = [alignPoints][2]float64{
	{38.2946, 51.6963}, // eye on the left of the picture
	{73.5318, 51.5014}, // eye on the right of the picture
	{56.0252, 71.7366}, // nose tip
	{56.1396, 92.2848}, // centre of the mouth
}

func (p *FacePipeline) faceEmbedding(frame *image.RGBA, det FaceDetection) ([]float32, error) {
	w, h, channels := tensorDimensions(p.embedInput)
	if channels != 3 || w <= 0 || h <= 0 {
		return nil, fmt.Errorf("unsupported embedding input %v", p.embedInput)
	}
	face := alignFace(frame, det, w, h, p.cfg.CropMargin)
	// SFace takes the colour values as they are (0 to 255), one colour plane after the other.
	input := pixelsToTensor(face, channelsFirst(p.embedInput), func(v uint8) float32 { return float32(v) })

	ctx, cancel := context.WithTimeout(context.Background(), inferTimeout)
	defer cancel()
	resp, err := p.embedModel.Infer(ctx, input, p.embedInput, p.embedDtype)
	if err != nil {
		return nil, err
	}
	emb := bytesToFloat32(resp.GetOutputData())
	if len(emb) == 0 {
		return nil, fmt.Errorf("embedding output is empty")
	}
	return normalizeL2(emb), nil
}

// alignFace returns the face as a w x h picture with the eyes, nose and mouth
// at the template positions. With no keypoints it is a plain square crop.
func alignFace(frame *image.RGBA, det FaceDetection, w, h int, margin float32) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	if det.hasKeypoints {
		if m, ok := faceTransform(det.keypoints, w, h); ok {
			xdraw.BiLinear.Transform(out, m, frame, frame.Bounds(), xdraw.Src, nil)
			return out
		}
	}
	cx := float64(det.XMin+det.XMax) / 2
	cy := float64(det.YMin+det.YMax) / 2
	half := float64(max(det.XMax-det.XMin, det.YMax-det.YMin)) * (1 + float64(margin)) / 2
	square := image.Rect(int(cx-half), int(cy-half), int(cx+half), int(cy+half)).Intersect(frame.Bounds())
	if !square.Empty() {
		xdraw.BiLinear.Scale(out, out.Bounds(), frame, square, xdraw.Src, nil)
	}
	return out
}

// faceTransform finds the rotation, scale and shift (no mirroring, no
// stretching) that best move the keypoints to the template positions.
func faceTransform(keypoints [alignPoints][2]float32, w, h int) (f64.Aff3, bool) {
	var src, dst [alignPoints][2]float64
	for i := range keypoints {
		src[i] = [2]float64{float64(keypoints[i][0]), float64(keypoints[i][1])}
		dst[i] = [2]float64{alignTemplate[i][0] * float64(w) / 112, alignTemplate[i][1] * float64(h) / 112}
	}
	m, residual := fitSimilarity(src, dst)

	// Which eye comes first depends on how the model names them. The right
	// order is the one that fits: with the eyes exchanged, the nose and the
	// mouth would have to land above them.
	src[0], src[1] = src[1], src[0]
	if swapped, r := fitSimilarity(src, dst); r < residual {
		m = swapped
	}

	scale := m[0]*m[0] + m[1]*m[1]
	if math.IsNaN(scale) || math.IsInf(scale, 0) || scale < 1e-8 {
		return f64.Aff3{}, false
	}
	return m, true
}

// fitSimilarity returns the least-squares similarity transform from src to
// dst, as the matrix that takes a source point to its place in the
// destination, and the squared error left.
func fitSimilarity(src, dst [alignPoints][2]float64) (f64.Aff3, float64) {
	var sx, sy, dx, dy float64
	for i := range src {
		sx += src[i][0]
		sy += src[i][1]
		dx += dst[i][0]
		dy += dst[i][1]
	}
	n := float64(len(src))
	sx, sy, dx, dy = sx/n, sy/n, dx/n, dy/n

	var dot, cross, norm float64
	for i := range src {
		px, py := src[i][0]-sx, src[i][1]-sy
		qx, qy := dst[i][0]-dx, dst[i][1]-dy
		dot += px*qx + py*qy
		cross += px*qy - py*qx
		norm += px*px + py*py
	}
	if norm < 1e-10 {
		return f64.Aff3{}, math.Inf(1)
	}
	a, b := dot/norm, cross/norm
	m := f64.Aff3{
		a, -b, dx - (a*sx - b*sy),
		b, a, dy - (b*sx + a*sy),
	}

	var residual float64
	for i := range src {
		ex := m[0]*src[i][0] + m[1]*src[i][1] + m[2] - dst[i][0]
		ey := m[3]*src[i][0] + m[4]*src[i][1] + m[5] - dst[i][1]
		residual += ex*ex + ey*ey
	}
	return m, residual
}

// ---------------------------------------------------------------- helpers

// decodeFrame decodes a camera frame into plain RGBA pixels, which the rest of
// the pipeline reads directly.
func decodeFrame(data []byte) (*image.RGBA, error) {
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode jpeg: %w", err)
	}
	if rgba, ok := img.(*image.RGBA); ok && rgba.Bounds().Min == (image.Point{}) {
		return rgba, nil
	}
	b := img.Bounds()
	if b.Empty() {
		return nil, fmt.Errorf("empty frame")
	}
	rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(rgba, rgba.Bounds(), img, b.Min, draw.Src)
	return rgba, nil
}

// resolveInputTensor returns the input shape and type the model declares, or
// the expected shape when the model does not give a complete one.
func resolveInputTensor(inputs []*aiv26.TensorInfo, expected []int32) ([]int32, aiv26.TensorDataType) {
	if len(inputs) == 0 {
		return expected, aiv26.TensorDataType_TENSOR_FLOAT32
	}
	shape := append([]int32(nil), inputs[0].GetShape()...)
	if len(shape) != len(expected) {
		return expected, inputs[0].GetDtype()
	}
	for i := range shape {
		if shape[i] <= 0 { // a size left open by the model
			shape[i] = expected[i]
		}
	}
	return shape, inputs[0].GetDtype()
}

// channelsFirst reports whether an input shape is (batch, channels, height,
// width), as ONNX models usually are, and not (batch, height, width, channels).
func channelsFirst(shape []int32) bool {
	return len(shape) == 4 && shape[1] == 3 && shape[3] != 3
}

// tensorDimensions reads width, height and channels from an input shape.
func tensorDimensions(shape []int32) (width, height, channels int) {
	switch {
	case channelsFirst(shape):
		return int(shape[3]), int(shape[2]), int(shape[1])
	case len(shape) == 4:
		return int(shape[2]), int(shape[1]), int(shape[3])
	case len(shape) == 3:
		return int(shape[1]), int(shape[0]), int(shape[2])
	default:
		return 0, 0, 0
	}
}

func bytesToFloat32(data []byte) []float32 {
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

func clamp(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}
