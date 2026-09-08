package facedetection

import (
	"context"
	"image"
	"image/color"
	"math"
	"testing"
)

func TestNewValidatesOptionsBeforeRuntimeAccess(t *testing.T) {
	tests := []struct {
		name    string
		library string
		model   string
		options Options
	}{
		{name: "missing runtime", model: "face.onnx"},
		{name: "missing model", library: "runtime.so"},
		{name: "negative threads", library: "runtime.so", model: "face.onnx", options: Options{IntraOpThreads: -1}},
		{name: "score too high", library: "runtime.so", model: "face.onnx", options: Options{ScoreThreshold: 1.1}},
		{name: "score NaN", library: "runtime.so", model: "face.onnx", options: Options{ScoreThreshold: math.NaN()}},
		{name: "NMS too high", library: "runtime.so", model: "face.onnx", options: Options{NMSThreshold: 1.1}},
		{name: "negative top-k", library: "runtime.so", model: "face.onnx", options: Options{TopK: -1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model, err := NewWithOptions(test.library, test.model, test.options)
			if model != nil || err == nil {
				t.Fatalf("NewWithOptions() = %v, %v; want validation error", model, err)
			}
		})
	}
}

func TestDetectRejectsCancelledContextBeforeRuntimeAccess(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := (&Model{}).Detect(ctx, image.NewNRGBA(image.Rect(0, 0, 1, 1)))
	if err != context.Canceled {
		t.Fatalf("Detect() error = %v, want context.Canceled", err)
	}
}

func TestPrepareIntoLetterboxesBGRValues(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	for x := range 2 {
		img.SetNRGBA(x, 0, color.NRGBA{R: 10, G: 20, B: 30, A: 255})
	}
	tensor := make([]float32, 3*InputWidth*InputHeight)
	content := prepareInto(tensor, img)
	wantContent := image.Rect(0, 160, 640, 480)
	if content != wantContent {
		t.Fatalf("content = %v, want %v", content, wantContent)
	}
	index := 320*InputWidth + 320
	pixels := InputWidth * InputHeight
	if tensor[index] != 30 || tensor[pixels+index] != 20 || tensor[2*pixels+index] != 10 {
		t.Fatalf("BGR values = %v, %v, %v", tensor[index], tensor[pixels+index], tensor[2*pixels+index])
	}
	if tensor[0] != 0 || tensor[pixels] != 0 || tensor[2*pixels] != 0 {
		t.Fatal("letterbox padding was not cleared")
	}
}

func TestPostprocessMapsAndFiltersFace(t *testing.T) {
	const scalarValues = 8400
	values := make([]float32, 2*scalarValues+4*scalarValues)
	row, column := 30, 40
	index := row*80 + column
	values[index] = 0.81
	values[scalarValues+index] = 1
	boxOffset := 2*scalarValues + 4*index
	values[boxOffset+2] = float32(math.Log(80.0 / 8.0))
	values[boxOffset+3] = float32(math.Log(80.0 / 8.0))

	detections := postprocess(values, image.Rect(0, 140, 640, 500), 0.8, 0.3, 5000)
	if len(detections) != 1 {
		t.Fatalf("detections = %+v, want one", detections)
	}
	detection := detections[0]
	x, y := detection.Center()
	if math.Abs(x-0.5) > 1e-6 || math.Abs(y-(100.0/360.0)) > 1e-6 {
		t.Fatalf("center = (%v, %v)", x, y)
	}
	if math.Abs(detection.Confidence-0.9) > 1e-6 {
		t.Fatalf("confidence = %v, want 0.9", detection.Confidence)
	}
	if got := postprocess(values, image.Rect(0, 140, 640, 500), 0.91, 0.3, 5000); len(got) != 0 {
		t.Fatalf("high-threshold detections = %+v, want none", got)
	}
}

func TestPostprocessSuppressesOverlappingFaces(t *testing.T) {
	const scalarValues = 8400
	values := make([]float32, 2*scalarValues+4*scalarValues)
	for column, score := range map[int]float32{40: 1, 41: 0.81} {
		index := 30*80 + column
		values[index] = score
		values[scalarValues+index] = 1
		boxOffset := 2*scalarValues + 4*index
		values[boxOffset+2] = float32(math.Log(80.0 / 8.0))
		values[boxOffset+3] = float32(math.Log(80.0 / 8.0))
	}
	if got := postprocess(values, image.Rect(0, 0, 640, 640), 0.8, 0.3, 5000); len(got) != 1 {
		t.Fatalf("detections = %+v, want one after NMS", got)
	}
}

func TestPrimaryFavorsProminentFace(t *testing.T) {
	small := Detection{Bounds: Box{MinX: 0, MinY: 0, MaxX: 0.1, MaxY: 0.1}, Confidence: 0.99}
	large := Detection{Bounds: Box{MinX: 0.3, MinY: 0.2, MaxX: 0.6, MaxY: 0.6}, Confidence: 0.85}
	got, ok := Primary([]Detection{small, large})
	if !ok || got != large {
		t.Fatalf("Primary() = %+v, %v; want prominent face", got, ok)
	}
	if _, ok := Primary(nil); ok {
		t.Fatal("Primary(nil) reported a face")
	}
}
