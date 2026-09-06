// Package saliency runs salient-object detection with ONNX Runtime.
package saliency

import (
	"errors"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	InputWidth  = 320
	InputHeight = 320
)

// Model owns a single reusable ONNX Runtime session.
type Model struct {
	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
	closed  bool
}

// New initializes ONNX Runtime and loads the model once.
func New(runtimeLibraryPath, modelPath string) (*Model, error) {
	if runtimeLibraryPath == "" {
		return nil, errors.New("ONNX Runtime library path is required")
	}
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	ort.SetSharedLibraryPath(runtimeLibraryPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime: %w", err)
	}

	// U2-Net exposes seven side outputs. The first (1959) is the fused,
	// highest-resolution saliency map and is the only output needed here.
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input.1"},
		[]string{"1959"},
		nil,
	)
	if err != nil {
		_ = ort.DestroyEnvironment()
		return nil, fmt.Errorf("load saliency model: %w", err)
	}

	return &Model{session: session}, nil
}

// Infer returns the model's 320x320 fused saliency map.
func (m *Model) Infer(input []float32) ([]float32, error) {
	if len(input) != 3*InputWidth*InputHeight {
		return nil, fmt.Errorf("invalid input tensor length: got %d", len(input))
	}

	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, InputHeight, InputWidth),
		input,
	)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](
		ort.NewShape(1, 1, InputHeight, InputWidth),
	)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, errors.New("saliency model is closed")
	}
	if err := m.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	); err != nil {
		return nil, fmt.Errorf("run saliency model: %w", err)
	}

	result := make([]float32, InputWidth*InputHeight)
	copy(result, outputTensor.GetData())
	return result, nil
}

// Close releases the model session and ONNX Runtime environment.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	var closeErrors []error
	if err := m.session.Destroy(); err != nil {
		closeErrors = append(closeErrors, fmt.Errorf("destroy model session: %w", err))
	}
	if err := ort.DestroyEnvironment(); err != nil {
		closeErrors = append(closeErrors, fmt.Errorf("destroy ONNX Runtime: %w", err))
	}
	return errors.Join(closeErrors...)
}
