package focalnet

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestNewRequiresPaths(t *testing.T) {
	for _, tt := range []struct{ name, library, model, want string }{
		{"missing runtime", "", "model.onnx", "ONNX Runtime library path is required"},
		{"missing model", "runtime.so", "", "model path is required"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model, err := New(tt.library, tt.model)
			if model != nil || err == nil || err.Error() != tt.want {
				t.Fatalf("New() = %v, %v; want %q", model, err, tt.want)
			}
		})
	}
}

func TestNewRejectsNegativeIntraOpThreadsBeforeRuntimeAccess(t *testing.T) {
	model, err := NewWithOptions("runtime.so", "model.onnx", Options{IntraOpThreads: -1})
	if model != nil || err == nil || err.Error() != "ONNX intra-op threads must be at least zero" {
		t.Fatalf("NewWithOptions() = %v, %v", model, err)
	}
}

func TestInferRejectsCancelledContextBeforeRuntimeAccess(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := (&Model{}).Infer(
		ctx,
		make([]float32, 3*InputSize*InputSize),
		make([]float32, MaxCandidates*4),
		make([]float32, 4),
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Infer() error = %v, want context.Canceled", err)
	}
}

func TestInferRejectsInvalidTensorBeforeRuntimeAccess(t *testing.T) {
	model := &Model{}
	image := make([]float32, 3*InputSize*InputSize)
	boxes := make([]float32, MaxCandidates*4)
	content := make([]float32, 4)
	for _, length := range []int{0, 1, InputSize * InputSize, 3*InputSize*InputSize - 1, 3*InputSize*InputSize + 1} {
		importance, scores, err := model.Infer(context.Background(), make([]float32, length), boxes, content)
		if importance != nil || scores != nil || err == nil || !strings.Contains(err.Error(), "invalid input tensor length") {
			t.Fatalf("Infer(image %d) = %v, %v, %v", length, importance, scores, err)
		}
	}
	importance, scores, err := model.Infer(context.Background(), image, make([]float32, 4), content)
	if importance != nil || scores != nil || err == nil || !strings.Contains(err.Error(), "invalid boxes tensor length") {
		t.Fatalf("Infer(short boxes) = %v, %v, %v", importance, scores, err)
	}
	importance, scores, err = model.Infer(context.Background(), image, boxes, make([]float32, 3))
	if importance != nil || scores != nil || err == nil || !strings.Contains(err.Error(), "invalid content tensor length") {
		t.Fatalf("Infer(short content) = %v, %v, %v", importance, scores, err)
	}
}
