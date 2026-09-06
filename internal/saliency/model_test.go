package saliency

import (
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

func TestInferRejectsInvalidTensorBeforeRuntimeAccess(t *testing.T) {
	model := &Model{}
	for _, length := range []int{0, 1, InputWidth * InputHeight, 3*InputWidth*InputHeight - 1, 3*InputWidth*InputHeight + 1} {
		output, err := model.Infer(make([]float32, length))
		if output != nil || err == nil || !strings.Contains(err.Error(), "invalid input tensor length") {
			t.Fatalf("Infer(%d values) = %v, %v", length, output, err)
		}
	}
}
