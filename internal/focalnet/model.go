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

// New initializes ONNX Runtime and loads a format-v2 human-ranking model.
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
		[]string{"image", "boxes", "content"},
		[]string{"importance", "crop_scores"},
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("load focalnet model: %w", err)
	}

	loaded = true
	return &Model{session: session, environmentHeld: true}, nil
}

// Infer runs the published human-ranking graph. image is NCHW 256×256, boxes
// is 128×4, and content is the letterbox rectangle [left, top, right, bottom].
func (m *Model) Infer(ctx context.Context, image, boxes, content []float32) (importance, scores []float32, err error) {
	if ctx == nil {
		return nil, nil, errors.New("inference context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if len(image) != 3*InputSize*InputSize {
		return nil, nil, fmt.Errorf("invalid input tensor length: got %d", len(image))
	}
	if len(boxes) != MaxCandidates*4 {
		return nil, nil, fmt.Errorf("invalid boxes tensor length: got %d", len(boxes))
	}
	if len(content) != 4 {
		return nil, nil, fmt.Errorf("invalid content tensor length: got %d", len(content))
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, nil, errors.New("focalnet model is closed")
	}

	imageTensor, err := ort.NewTensor(ort.NewShape(1, 3, InputSize, InputSize), image)
	if err != nil {
		return nil, nil, fmt.Errorf("create image tensor: %w", err)
	}
	defer imageTensor.Destroy()
	boxesTensor, err := ort.NewTensor(ort.NewShape(1, MaxCandidates, 4), boxes)
	if err != nil {
		return nil, nil, fmt.Errorf("create boxes tensor: %w", err)
	}
	defer boxesTensor.Destroy()
	contentTensor, err := ort.NewTensor(ort.NewShape(1, 4), content)
	if err != nil {
		return nil, nil, fmt.Errorf("create content tensor: %w", err)
	}
	defer contentTensor.Destroy()

	importance = make([]float32, MapSize*MapSize)
	importanceTensor, err := ort.NewTensor(ort.NewShape(1, 1, MapSize, MapSize), importance)
	if err != nil {
		return nil, nil, fmt.Errorf("create importance tensor: %w", err)
	}
	defer importanceTensor.Destroy()
	scores = make([]float32, MaxCandidates)
	scoresTensor, err := ort.NewTensor(ort.NewShape(1, MaxCandidates), scores)
	if err != nil {
		return nil, nil, fmt.Errorf("create scores tensor: %w", err)
	}
	defer scoresTensor.Destroy()

	runOptions, err := ort.NewRunOptions()
	if err != nil {
		return nil, nil, fmt.Errorf("create inference run options: %w", err)
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
		[]ort.Value{imageTensor, boxesTensor, contentTensor},
		[]ort.Value{importanceTensor, scoresTensor},
		runOptions,
	)
	close(stopWatcher)
	<-watcherDone
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, nil, ctxErr
	}
	if runErr != nil {
		return nil, nil, fmt.Errorf("run focalnet model: %w", runErr)
	}
	if err := validateImportance(importance); err != nil {
		return nil, nil, err
	}
	for i, value := range importance {
		if value < 0 {
			importance[i] = 0
		} else if value > 1 {
			importance[i] = 1
		}
	}
	for _, value := range scores {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, errors.New("model produced invalid crop scores")
		}
	}
	return importance, scores, nil
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
