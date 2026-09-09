// Package facedetection finds faces with the YuNet ONNX model.
package facedetection

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"sort"
	"sync"

	"autogravity/internal/ortenv"
	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	InputWidth            = 640
	InputHeight           = 640
	DefaultScoreThreshold = 0.85
	DefaultNMSThreshold   = 0.30
	DefaultTopK           = 5000
)

var (
	outputNames = []string{
		"cls_8", "cls_16", "cls_32",
		"obj_8", "obj_16", "obj_32",
		"bbox_8", "bbox_16", "bbox_32",
	}
	outputShapes = []ort.Shape{
		ort.NewShape(1, 6400, 1), ort.NewShape(1, 1600, 1), ort.NewShape(1, 400, 1),
		ort.NewShape(1, 6400, 1), ort.NewShape(1, 1600, 1), ort.NewShape(1, 400, 1),
		ort.NewShape(1, 6400, 4), ort.NewShape(1, 1600, 4), ort.NewShape(1, 400, 4),
	}
)

// Box is a normalized face rectangle in oriented-image coordinates.
type Box struct {
	MinX float64
	MinY float64
	MaxX float64
	MaxY float64
}

// Detection is one face candidate retained after score filtering and NMS.
type Detection struct {
	Bounds     Box
	Confidence float64
}

// Center returns the normalized center of the detected face.
func (d Detection) Center() (float64, float64) {
	return (d.Bounds.MinX + d.Bounds.MaxX) / 2, (d.Bounds.MinY + d.Bounds.MaxY) / 2
}

// Primary chooses the most prominent reliable face. Area favors foreground
// faces while confidence prevents a large, weak candidate from dominating.
func Primary(detections []Detection) (Detection, bool) {
	if len(detections) == 0 {
		return Detection{}, false
	}
	best := detections[0]
	bestPriority := priority(best)
	for _, detection := range detections[1:] {
		candidatePriority := priority(detection)
		if candidatePriority > bestPriority ||
			(candidatePriority == bestPriority && detection.Confidence > best.Confidence) {
			best = detection
			bestPriority = candidatePriority
		}
	}
	return best, true
}

func priority(detection Detection) float64 {
	width := math.Max(0, detection.Bounds.MaxX-detection.Bounds.MinX)
	height := math.Max(0, detection.Bounds.MaxY-detection.Bounds.MinY)
	return detection.Confidence * math.Sqrt(width*height)
}

// Options controls face filtering and the ONNX Runtime session.
type Options struct {
	ScoreThreshold float64
	NMSThreshold   float64
	TopK           int
	IntraOpThreads int
}

// Model owns a reusable YuNet session and its preprocessing buffers.
type Model struct {
	mu              sync.RWMutex
	session         *ort.DynamicAdvancedSession
	environmentHeld bool
	closed          bool
	scoreThreshold  float64
	nmsThreshold    float64
	topK            int
	inputBuffers    sync.Pool
}

// New initializes a face detector with production defaults.
func New(runtimeLibraryPath, modelPath string) (*Model, error) {
	return NewWithOptions(runtimeLibraryPath, modelPath, Options{})
}

// NewWithOptions initializes a face detector with explicit filtering and
// session settings.
func NewWithOptions(runtimeLibraryPath, modelPath string, options Options) (*Model, error) {
	if runtimeLibraryPath == "" {
		return nil, errors.New("ONNX Runtime library path is required")
	}
	if modelPath == "" {
		return nil, errors.New("face model path is required")
	}
	if options.IntraOpThreads < 0 {
		return nil, errors.New("ONNX intra-op threads must be at least zero")
	}
	if options.ScoreThreshold == 0 {
		options.ScoreThreshold = DefaultScoreThreshold
	}
	if options.NMSThreshold == 0 {
		options.NMSThreshold = DefaultNMSThreshold
	}
	if options.TopK == 0 {
		options.TopK = DefaultTopK
	}
	if math.IsNaN(options.ScoreThreshold) || math.IsInf(options.ScoreThreshold, 0) ||
		options.ScoreThreshold <= 0 || options.ScoreThreshold > 1 {
		return nil, errors.New("face score threshold must be greater than zero and at most one")
	}
	if math.IsNaN(options.NMSThreshold) || math.IsInf(options.NMSThreshold, 0) ||
		options.NMSThreshold <= 0 || options.NMSThreshold > 1 {
		return nil, errors.New("face NMS threshold must be greater than zero and at most one")
	}
	if options.TopK < 1 {
		return nil, errors.New("face top-k must be positive")
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
		return nil, fmt.Errorf("create ONNX session options: %w", err)
	}
	defer sessionOptions.Destroy()
	if err := sessionOptions.SetIntraOpNumThreads(options.IntraOpThreads); err != nil {
		return nil, fmt.Errorf("configure ONNX intra-op threads: %w", err)
	}
	if err := sessionOptions.SetExecutionMode(ort.ExecutionModeSequential); err != nil {
		return nil, fmt.Errorf("configure ONNX execution mode: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input"},
		outputNames,
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("load face model: %w", err)
	}

	loaded = true
	model := &Model{
		session:         session,
		environmentHeld: true,
		scoreThreshold:  options.ScoreThreshold,
		nmsThreshold:    options.NMSThreshold,
		topK:            options.TopK,
	}
	model.inputBuffers.New = func() any {
		return make([]float32, 3*InputWidth*InputHeight)
	}
	return model, nil
}

// Detect returns normalized face boxes from an oriented image.
func (m *Model) Detect(ctx context.Context, img image.Image) ([]Detection, error) {
	if ctx == nil {
		return nil, errors.New("inference context is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if img == nil || img.Bounds().Dx() <= 0 || img.Bounds().Dy() <= 0 {
		return nil, errors.New("invalid image")
	}

	input := m.inputBuffers.Get().([]float32)
	defer m.inputBuffers.Put(input)
	content := prepareInto(input, img)
	outputs, err := m.infer(ctx, input)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return postprocess(outputs, content, m.scoreThreshold, m.nmsThreshold, m.topK), nil
}

func (m *Model) infer(ctx context.Context, input []float32) ([]float32, error) {
	if len(input) != 3*InputWidth*InputHeight {
		return nil, fmt.Errorf("invalid input tensor length: got %d", len(input))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, errors.New("face model is closed")
	}

	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, InputHeight, InputWidth),
		input,
	)
	if err != nil {
		return nil, fmt.Errorf("create face input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputData := make([][]float32, len(outputShapes))
	outputValues := make([]ort.Value, len(outputShapes))
	for i, shape := range outputShapes {
		outputData[i] = make([]float32, int(shape.FlattenedSize()))
		tensor, tensorErr := ort.NewTensor(shape, outputData[i])
		if tensorErr != nil {
			for _, value := range outputValues[:i] {
				_ = value.Destroy()
			}
			return nil, fmt.Errorf("create face output tensor: %w", tensorErr)
		}
		outputValues[i] = tensor
	}
	defer func() {
		for _, value := range outputValues {
			_ = value.Destroy()
		}
	}()

	runOptions, err := ort.NewRunOptions()
	if err != nil {
		return nil, fmt.Errorf("create face run options: %w", err)
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
	runErr := m.session.RunWithOptions([]ort.Value{inputTensor}, outputValues, runOptions)
	close(stopWatcher)
	<-watcherDone
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}
	if runErr != nil {
		return nil, fmt.Errorf("run face model: %w", runErr)
	}

	flattened := make([]float32, 0, 50_400)
	for _, values := range outputData {
		flattened = append(flattened, values...)
	}
	return flattened, nil
}

func prepareInto(tensor []float32, img image.Image) image.Rectangle {
	clear(tensor)
	sourceWidth, sourceHeight := img.Bounds().Dx(), img.Bounds().Dy()
	scale := math.Min(float64(InputWidth)/float64(sourceWidth), float64(InputHeight)/float64(sourceHeight))
	resizedWidth := max(1, min(InputWidth, int(math.Round(float64(sourceWidth)*scale))))
	resizedHeight := max(1, min(InputHeight, int(math.Round(float64(sourceHeight)*scale))))
	offsetX := (InputWidth - resizedWidth) / 2
	offsetY := (InputHeight - resizedHeight) / 2
	content := image.Rect(offsetX, offsetY, offsetX+resizedWidth, offsetY+resizedHeight)
	resized := imaging.Resize(img, resizedWidth, resizedHeight, imaging.Linear)
	pixels := InputWidth * InputHeight
	for y := 0; y < resizedHeight; y++ {
		row := resized.Pix[y*resized.Stride : y*resized.Stride+4*resizedWidth]
		for x := 0; x < resizedWidth; x++ {
			pixel := row[4*x : 4*x+4]
			alpha := uint32(pixel[3])
			index := (y+offsetY)*InputWidth + x + offsetX
			// YuNet follows OpenCV's BGR channel order and consumes raw 0..255 values.
			tensor[index] = float32(uint32(pixel[2]) * alpha / 255)
			tensor[pixels+index] = float32(uint32(pixel[1]) * alpha / 255)
			tensor[2*pixels+index] = float32(uint32(pixel[0]) * alpha / 255)
		}
	}
	return content
}

type candidate struct {
	x1, y1, x2, y2 float64
	score          float64
}

func postprocess(flattened []float32, content image.Rectangle, scoreThreshold, nmsThreshold float64, topK int) []Detection {
	const scalarValues = 6400 + 1600 + 400
	const boxValues = scalarValues * 4
	if len(flattened) != 2*scalarValues+boxValues || content.Empty() {
		return nil
	}
	classValues := flattened[:scalarValues]
	objectValues := flattened[scalarValues : 2*scalarValues]
	boxValuesSlice := flattened[2*scalarValues:]

	strides := [...]int{8, 16, 32}
	counts := [...]int{6400, 1600, 400}
	var candidates []candidate
	offset := 0
	boxOffset := 0
	for level, stride := range strides {
		columns := InputWidth / stride
		for index := 0; index < counts[level]; index++ {
			classScore := clamp01(float64(classValues[offset+index]))
			objectScore := clamp01(float64(objectValues[offset+index]))
			score := math.Sqrt(classScore * objectScore)
			if score < scoreThreshold || math.IsNaN(score) {
				continue
			}
			row, column := index/columns, index%columns
			values := boxValuesSlice[boxOffset+4*index : boxOffset+4*index+4]
			centerX := (float64(column) + float64(values[0])) * float64(stride)
			centerY := (float64(row) + float64(values[1])) * float64(stride)
			width := math.Exp(float64(values[2])) * float64(stride)
			height := math.Exp(float64(values[3])) * float64(stride)
			if width <= 0 || height <= 0 || math.IsNaN(width) || math.IsNaN(height) ||
				math.IsInf(width, 0) || math.IsInf(height, 0) || math.IsNaN(centerX) ||
				math.IsNaN(centerY) || math.IsInf(centerX, 0) || math.IsInf(centerY, 0) {
				continue
			}
			candidates = append(candidates, candidate{
				x1: centerX - width/2, y1: centerY - height/2,
				x2: centerX + width/2, y2: centerY + height/2,
				score: score,
			})
		}
		offset += counts[level]
		boxOffset += counts[level] * 4
	}

	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}
	kept := make([]candidate, 0, len(candidates))
	for _, current := range candidates {
		suppressed := false
		for _, existing := range kept {
			if intersectionOverUnion(current, existing) >= nmsThreshold {
				suppressed = true
				break
			}
		}
		if !suppressed {
			kept = append(kept, current)
		}
	}

	contentMinX, contentMinY := float64(content.Min.X), float64(content.Min.Y)
	contentMaxX, contentMaxY := float64(content.Max.X), float64(content.Max.Y)
	contentWidth, contentHeight := float64(content.Dx()), float64(content.Dy())
	detections := make([]Detection, 0, len(kept))
	for _, face := range kept {
		x1 := math.Max(face.x1, contentMinX)
		y1 := math.Max(face.y1, contentMinY)
		x2 := math.Min(face.x2, contentMaxX)
		y2 := math.Min(face.y2, contentMaxY)
		if x2 <= x1 || y2 <= y1 {
			continue
		}
		detections = append(detections, Detection{
			Bounds: Box{
				MinX: clamp01((x1 - contentMinX) / contentWidth),
				MinY: clamp01((y1 - contentMinY) / contentHeight),
				MaxX: clamp01((x2 - contentMinX) / contentWidth),
				MaxY: clamp01((y2 - contentMinY) / contentHeight),
			},
			Confidence: face.score,
		})
	}
	return detections
}

func intersectionOverUnion(a, b candidate) float64 {
	intersectionWidth := math.Max(0, math.Min(a.x2, b.x2)-math.Max(a.x1, b.x1))
	intersectionHeight := math.Max(0, math.Min(a.y2, b.y2)-math.Max(a.y1, b.y1))
	intersection := intersectionWidth * intersectionHeight
	areaA := math.Max(0, a.x2-a.x1) * math.Max(0, a.y2-a.y1)
	areaB := math.Max(0, b.x2-b.x1) * math.Max(0, b.y2-b.y1)
	union := areaA + areaB - intersection
	if union <= 0 {
		return 0
	}
	return intersection / union
}

func clamp01(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

// Close releases the face session and its shared runtime reference.
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
			closeErrors = append(closeErrors, fmt.Errorf("destroy face model session: %w", err))
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
