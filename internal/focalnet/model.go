package focalnet

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"autogravity/internal/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

// Model owns a single reusable FocalNet ONNX Runtime session.
type Model struct {
	mu              sync.RWMutex
	session         *ort.DynamicAdvancedSession
	environmentHeld bool
	closed          bool
}

// Options controls the ONNX Runtime session used by the model.
type Options struct {
	IntraOpThreads int
}

// New initializes ONNX Runtime and loads a format-v1 importance model.
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
	if err := sessionOptions.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
		return nil, fmt.Errorf("configure ONNX Runtime execution mode: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"image"},
		[]string{"importance"},
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("load focalnet model: %w", err)
	}

	loaded = true
	return &Model{session: session, environmentHeld: true}, nil
}

// Infer returns the model's 64x64 letterboxed importance map.
func (m *Model) Infer(ctx context.Context, input []float32) ([]float32, error) {
	if ctx == nil {
		return nil, errors.New("inference context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(input) != 3*InputSize*InputSize {
		return nil, fmt.Errorf("invalid input tensor length: got %d", len(input))
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, errors.New("focalnet model is closed")
	}

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, InputSize, InputSize), input)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	result := make([]float32, MapSize*MapSize)
	outputTensor, err := ort.NewTensor(ort.NewShape(1, 1, MapSize, MapSize), result)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	runOptions, err := ort.NewRunOptions()
	if err != nil {
		return nil, fmt.Errorf("create inference run options: %w", err)
	}
	defer runOptions.Destroy()

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
	<-watcherDone
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}
	if runErr != nil {
		return nil, fmt.Errorf("run focalnet model: %w", runErr)
	}
	if err := validateImportance(result); err != nil {
		return nil, err
	}
	for i, value := range result {
		if value < 0 {
			result[i] = 0
		} else if value > 1 {
			result[i] = 1
		}
	}
	return result, nil
}

func validateImportance(values []float32) error {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < -1e-5 || value > 1.00001 {
			return errors.New("model produced an invalid importance map")
		}
	}
	return nil
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
