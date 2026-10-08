package main

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"testing"
	"time"
)

func near(a, b, tolerance float64) bool { return math.Abs(a-b) <= tolerance }

// A 640x480 frame in a 192x192 input: scaled by 0.3, with bars of 24 pixels above and below.
func TestLetterbox(t *testing.T) {
	box, inside := fitLetterbox(640, 480, 192, 192)
	if !near(box.scale, 0.3, 1e-9) || box.padX != 0 || box.padY != 24 {
		t.Fatalf("letterbox = %+v", box)
	}
	if inside != image.Rect(0, 24, 192, 168) {
		t.Fatalf("picture placed at %v", inside)
	}
	if x, y := box.toFrame(96, 96); !near(float64(x), 320, 0.01) || !near(float64(y), 240, 0.01) {
		t.Fatalf("centre maps to %v, %v", x, y)
	}
	if _, y := box.toFrame(0, 24); !near(float64(y), 0, 0.01) {
		t.Fatalf("top of the picture maps to y=%v", y)
	}
}

// The scaled picture is the average of the frame, and the bars stay black.
func TestScaleInto(t *testing.T) {
	frame := image.NewRGBA(image.Rect(0, 0, 640, 480))
	for y := 0; y < 480; y++ {
		for x := 0; x < 640; x++ {
			v := uint8(0)
			if x >= 320 {
				v = 200
			}
			frame.SetRGBA(x, y, color.RGBA{R: v, G: uint8(x % 2 * 100), B: 10, A: 255})
		}
	}
	canvas := image.NewRGBA(image.Rect(0, 0, 192, 192))
	_, inside := fitLetterbox(640, 480, 192, 192)
	scaleInto(canvas, inside, frame)
	if c := canvas.RGBAAt(10, 96); c.R != 0 || c.B != 10 || c.G < 30 || c.G > 70 { // a block is 3 or 4 columns wide
		t.Fatalf("left half: %v", c)
	}
	if c := canvas.RGBAAt(180, 96); c.R != 200 {
		t.Fatalf("right half: %v", c)
	}
	if c := canvas.RGBAAt(96, 5); c.R != 0 || c.G != 0 || c.B != 0 {
		t.Fatalf("bar: %v", c)
	}
}

func TestSigmoidAlwaysApplies(t *testing.T) {
	// A raw score of 0.3 is a probability of 0.574, not 0.3.
	if got := sigmoid(0.3); !near(float64(got), 0.5744, 0.001) {
		t.Fatalf("sigmoid(0.3) = %v", got)
	}
	if sigmoid(-1000) > 1e-6 || sigmoid(1000) < 1-1e-6 {
		t.Fatalf("extreme scores: %v %v", sigmoid(-1000), sigmoid(1000))
	}
}

func TestPixelsToTensor(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 0, G: 255, B: 51, A: 255})
	img.SetRGBA(1, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})
	out := bytesToFloat32(pixelsToTensor(img, false, func(v uint8) float32 { return float32(v)/127.5 - 1 }))
	if len(out) != 6 || out[0] != -1 || out[1] != 1 || !near(float64(out[2]), -0.6, 1e-6) || out[3] != 1 {
		t.Fatalf("pixel after pixel = %v", out)
	}
	// Plane after plane: the reds, then the greens, then the blues, with the values as they are.
	out = bytesToFloat32(pixelsToTensor(img, true, func(v uint8) float32 { return float32(v) }))
	if want := []float32{0, 255, 255, 0, 51, 0}; len(out) != 6 || out[0] != want[0] || out[1] != want[1] || out[2] != want[2] || out[3] != want[3] || out[4] != want[4] || out[5] != want[5] {
		t.Fatalf("plane after plane = %v", out)
	}
	if !channelsFirst([]int32{1, 3, 112, 112}) || channelsFirst([]int32{1, 192, 192, 3}) {
		t.Fatal("input layout not recognised")
	}
	if w, h, c := tensorDimensions([]int32{1, 3, 112, 96}); w != 96 || h != 112 || c != 3 {
		t.Fatalf("channels-first size = %d x %d x %d", w, h, c)
	}
}

// One strong detection on the anchor at the centre of the input.
func TestDecodeBlazeFace(t *testing.T) {
	const net = 192
	anchors := buildBlazeAnchors(2304, net, net)
	if len(anchors) != 2304 {
		t.Fatalf("%d anchors", len(anchors))
	}
	reg := make([]float32, 2304*16)
	cls := make([]float32, 2304)
	for i := range cls {
		cls[i] = -20
	}
	i := 24*48 + 24 // row 24, column 24: anchor centre at (98, 98) in model pixels
	cls[i] = 4
	cls[i+1] = 3 // the neighbour sees the same face, a little weaker
	for _, j := range []int{i, i + 1} {
		shift := float32(i-j) * 4 // both describe the same box
		r := reg[j*16:]
		r[0], r[1], r[2], r[3] = shift-2, -2, 30, 30 // centre (96, 96), 30 x 30
		r[4], r[5] = shift-8, -6                     // eye on the left
		r[6], r[7] = shift+4, -6                     // eye on the right
		r[8], r[9] = shift-2, 0                      // nose
		r[10], r[11] = shift-2, 6                    // mouth
	}
	tensors := []floatTensor{
		{Data: reg, Rows: 2304, Cols: 16, Valid: true},
		{Data: cls, Rows: 2304, Cols: 1, Valid: true},
	}
	box, _ := fitLetterbox(640, 480, net, net)
	dets := decodeBlazeFace(tensors, box, 640, 480, net, net, 0.5, 5)
	if len(dets) != 1 {
		t.Fatalf("%d detections, want 1 (the two anchors describe one face)", len(dets))
	}
	d := dets[0]
	// (96±15 - pad) / 0.3: x 270..370, y 190..290
	if d.XMin != 270 || d.XMax != 370 || d.YMin != 190 || d.YMax != 290 {
		t.Fatalf("box = %d,%d - %d,%d", d.XMin, d.YMin, d.XMax, d.YMax)
	}
	if !near(float64(d.Score), 0.982, 0.001) {
		t.Fatalf("score = %v", d.Score)
	}
	if !d.hasKeypoints || !near(float64(d.keypoints[0][0]), 300, 0.01) || !near(float64(d.keypoints[0][1]), 226.67, 0.01) ||
		!near(float64(d.keypoints[1][0]), 340, 0.01) || !near(float64(d.keypoints[3][1]), 266.67, 0.01) {
		t.Fatalf("keypoints = %v", d.keypoints)
	}

	// A weak raw score below the threshold is dropped; just above, it is kept.
	cls[i], cls[i+1] = -0.1, -20
	if n := len(decodeBlazeFace(tensors, box, 640, 480, net, net, 0.5, 5)); n != 0 {
		t.Fatalf("raw score -0.1: %d detections, want 0", n)
	}
	cls[i] = 0.1
	if n := len(decodeBlazeFace(tensors, box, 640, 480, net, net, 0.5, 5)); n != 1 {
		t.Fatalf("raw score 0.1: %d detections, want 1", n)
	}
}

// A face drawn tilted and at another size: the transform must bring its
// keypoints to the template, whichever eye is given first.
func TestFaceTransform(t *testing.T) {
	angle, scale, tx, ty := 0.4, 2.5, 300.0, 180.0
	var kp [alignPoints][2]float32
	for i, p := range alignTemplate {
		x := scale*(math.Cos(angle)*p[0]-math.Sin(angle)*p[1]) + tx
		y := scale*(math.Sin(angle)*p[0]+math.Cos(angle)*p[1]) + ty
		kp[i] = [2]float32{float32(x), float32(y)}
	}
	check := func(name string, kp [alignPoints][2]float32) {
		m, ok := faceTransform(kp, 112, 112)
		if !ok {
			t.Fatalf("%s: no transform", name)
		}
		eyeY := 0.0
		for i := 0; i < 2; i++ {
			eyeY += (m[3]*float64(kp[i][0]) + m[4]*float64(kp[i][1]) + m[5]) / 2
		}
		mouthX := m[0]*float64(kp[3][0]) + m[1]*float64(kp[3][1]) + m[2]
		mouthY := m[3]*float64(kp[3][0]) + m[4]*float64(kp[3][1]) + m[5]
		if !near(eyeY, 51.6, 0.2) || !near(mouthX, alignTemplate[3][0], 0.1) || !near(mouthY, alignTemplate[3][1], 0.1) {
			t.Fatalf("%s: eyes at y=%.1f, mouth at %.1f,%.1f", name, eyeY, mouthX, mouthY)
		}
	}
	check("eyes in order", kp)
	kp[0], kp[1] = kp[1], kp[0]
	check("eyes exchanged", kp)
}

// The aligned picture really has the eye where the template says.
func TestAlignFaceMovesPixels(t *testing.T) {
	frame := image.NewRGBA(image.Rect(0, 0, 640, 480))
	det := FaceDetection{XMin: 200, YMin: 100, XMax: 440, YMax: 340, hasKeypoints: true}
	for i, p := range alignTemplate { // the face is the template, twice the size, moved
		det.keypoints[i] = [2]float32{float32(2*p[0] + 208), float32(2*p[1] + 108)}
	}
	// A red patch on the eye at the left of the picture.
	ex, ey := int(det.keypoints[0][0]), int(det.keypoints[0][1])
	for y := ey - 6; y <= ey+6; y++ {
		for x := ex - 6; x <= ex+6; x++ {
			frame.SetRGBA(x, y, color.RGBA{R: 255, A: 255})
		}
	}
	face := alignFace(frame, det, 112, 112, 0.25)
	if c := face.RGBAAt(38, 52); c.R < 200 {
		t.Fatalf("left eye position is %v, want red", c)
	}
	if c := face.RGBAAt(74, 52); c.R > 50 {
		t.Fatalf("right eye position is %v, want dark", c)
	}

	// With no keypoints: a square crop around the box, still the right size.
	det.hasKeypoints = false
	if b := alignFace(frame, det, 112, 112, 0.25).Bounds(); b.Dx() != 112 || b.Dy() != 112 {
		t.Fatalf("crop size %v", b)
	}
}

func TestFollowKeepsNumbers(t *testing.T) {
	p := &FacePipeline{}
	now := time.Now()
	a := FaceDetection{XMin: 100, YMin: 100, XMax: 200, YMax: 200}
	b := FaceDetection{XMin: 400, YMin: 100, XMax: 500, YMax: 200}
	first := p.follow([]FaceDetection{a, b}, now)
	idA, idB := first[0].id, first[1].id
	if idA == idB {
		t.Fatal("two faces share a number")
	}
	// Both move a little and arrive in the other order.
	a.XMin, a.XMax = 110, 210
	b.XMin, b.XMax = 390, 490
	second := p.follow([]FaceDetection{b, a}, now.Add(100*time.Millisecond))
	if second[0].id != idB || second[1].id != idA {
		t.Fatalf("numbers changed: %d,%d then %d,%d", idA, idB, second[1].id, second[0].id)
	}
	// A face that comes back after a long time is a new face.
	third := p.follow([]FaceDetection{a}, now.Add(5*time.Second))
	if third[0].id == idA || len(p.tracks) != 1 {
		t.Fatalf("after 5 s: number %d, %d tracks", third[0].id, len(p.tracks))
	}
}

func TestDecodeFrame(t *testing.T) {
	src := image.NewYCbCr(image.Rect(0, 0, 64, 48), image.YCbCrSubsampleRatio420)
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, nil); err != nil {
		t.Fatal(err)
	}
	frame, err := decodeFrame(buf.Bytes())
	if err != nil || frame.Bounds() != image.Rect(0, 0, 64, 48) {
		t.Fatalf("frame %v, %v", frame, err)
	}
	if _, err := decodeFrame([]byte("not a picture")); err == nil {
		t.Fatal("no error for bad data")
	}
}

func BenchmarkPrepareDetectorInput(b *testing.B) {
	frame := image.NewRGBA(image.Rect(0, 0, 640, 480))
	for i := range frame.Pix {
		frame.Pix[i] = uint8(i)
	}
	p := &FacePipeline{detectCanvas: image.NewRGBA(image.Rect(0, 0, 192, 192))}
	_, inside := fitLetterbox(640, 480, 192, 192)
	for i := 0; i < b.N; i++ {
		scaleInto(p.detectCanvas, inside, frame)
		_ = pixelsToTensor(p.detectCanvas, false, func(v uint8) float32 { return float32(v)/127.5 - 1 })
	}
}
