// Package saliency runs salient-object detection with ONNX Runtime.
package saliency

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"autogravity/internal/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	InputWidth  = 320
	InputHeight = 320
)

// Model owns a single reusable ONNX Runtime session.
type Model struct {
	mu              sync.RWMutex
	session         *ort.DynamicAdvancedSession
	environmentHeld bool
	closed          bool
}

// Options controls the ONNX Runtime session used by the model.
type Options struct {
	// IntraOpThreads is the number of threads used within each operator. Zero
	// keeps ONNX Runtime's default. Use one when running multiple requests in
	// parallel on a CPU-constrained container to avoid oversubscription.
	IntraOpThreads int
}

// New initializes ONNX Runtime and loads the model once.
func New(runtimeLibraryPath, modelPath string) (*Model, error) {
	return NewWithOptions(runtimeLibraryPath, modelPath, Options{})
}

// NewWithOptions initializes ONNX Runtime and loads the model once with the
// supplied session settings.
func NewWithOptions(runtimeLibraryPath, modelPath string, options Options) (*Model, error) {
	if runtimeLibraryPath == "" {
		return nil, errors.New("ONNX Runtime library path is required")
	}
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}
	if options.IntraOpThreads < 0 {
		return nil, errors.New("ONNX intra-op threads must be at least zero")
	}

	if err := ortenv.Acquire(runtimeLibraryPath); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime: %w", err)
	}
	loaded := false
	defer func() {
		if !loaded {
			// Registered before sessionOptions.Destroy: options must be freed
			// before the final reference unloads the shared library.
			_ = ortenv.Release()
		}
	}()

	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create ONNX Runtime session options: %w", err)
	}
	defer sessionOptions.Destroy()
	if err := sessionOptions.SetIntraOpNumThreads(options.IntraOpThreads); err != nil {
		return nil, fmt.Errorf("configure ONNX Runtime intra-op threads: %w", err)
	}
	// Sequential mode avoids creating an inter-op thread pool. Independent
	// requests still run concurrently by calling Run on the shared session.
	if err := sessionOptions.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
		return nil, fmt.Errorf("configure ONNX Runtime execution mode: %w", err)
	}

	// U2-Net exposes seven side outputs. The first (1959) is the fused,
	// highest-resolution saliency map and is the only output needed here.
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input.1"},
		[]string{"1959"},
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("load saliency model: %w", err)
	}

	loaded = true
	return &Model{session: session, environmentHeld: true}, nil
}

// Infer returns the model's 320x320 fused saliency map.
func (m *Model) Infer(ctx context.Context, input []float32) ([]float32, error) {
	if ctx == nil {
		return nil, errors.New("inference context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(input) != 3*InputWidth*InputHeight {
		return nil, fmt.Errorf("invalid input tensor length: got %d", len(input))
	}
	// ONNX Runtime supports concurrent Run calls on one CPU session. Hold the
	// lifecycle read lock across all runtime calls so Close cannot destroy the
	// environment while tensors or inference are active.
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, errors.New("saliency model is closed")
	}

	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, InputHeight, InputWidth),
		input,
	)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Keep ownership of the Go output buffer: destroying the ONNX tensor
	// releases the runtime wrapper, not this backing slice.
	result := make([]float32, InputWidth*InputHeight)
	outputTensor, err := ort.NewTensor(
		ort.NewShape(1, 1, InputHeight, InputWidth),
		result,
	)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	runOptions, err := ort.NewRunOptions()
	if err != nil {
		return nil, fmt.Errorf("create inference run options: %w", err)
	}
	defer runOptions.Destroy()

	// RunOptions belongs to this inference call. A shared instance would let a
	// cancelled request terminate unrelated concurrent requests.
	watcherDone := make(chan struct{})
	stopWatcher := make(chan struct{})
	go func() {
		defer close(watcherDone)
		select {
		case <-ctx.Done():
			_ = runOptions.Terminate()
		case <-stopWatcher:
		}
	}()

	runErr := m.session.RunWithOptions(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		runOptions,
	)
	close(stopWatcher)
	<-watcherDone // Ensure Terminate cannot race with Destroy.
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}
	if runErr != nil {
		return nil, fmt.Errorf("run saliency model: %w", runErr)
	}

	return result, nil
}

// Close releases the model session and its shared ONNX Runtime reference.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	var closeErrors []error
	if m.session != nil {
		if err := m.session.Destroy(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("destroy model session: %w", err))
		}
	}
	if m.environmentHeld {
		m.environmentHeld = false
		if err := ortenv.Release(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("release ONNX Runtime: %w", err))
		}
	}
	return errors.Join(closeErrors...)
}
