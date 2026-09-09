//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"autogravity/internal/facedetection"
	"autogravity/internal/imageutil"
	"autogravity/internal/saliency"
	"autogravity/internal/testimages"
	"github.com/disintegration/imaging"
)

// subjectCase bounds are visually chosen acceptable centroids in image coordinates.
type subjectCase struct {
	name                   string
	minX, maxX, minY, maxY float64
}

func TestAnalyzeRealModel(t *testing.T) {
	runSubjectCases(t, []subjectCase{
		{"rose.png", 0.25, 0.47, 0.30, 0.70},
		{"rose-lossless.webp", 0.25, 0.47, 0.30, 0.70},
		{"rose-lossy.webp", 0.25, 0.47, 0.30, 0.70},
		{"rose-alpha.webp", 0.25, 0.47, 0.30, 0.70},
		{"portrait.jpg", 0.35, 0.65, 0.30, 0.70},
		{"dog-portrait.jpg", 0.25, 0.75, 0.58, 0.85},
		{"puppies.jpg", 0.30, 0.70, 0.35, 0.75},
		{"panda-bamboo.jpg", 0.20, 0.80, 0.25, 0.85},
	})
}

// facePriorityCase bounds are visually annotated around the intended primary
// face before inference. Cases without a visible face exercise saliency fallback.
type facePriorityCase struct {
	name                   string
	wantSource             string
	minX, maxX, minY, maxY float64
}

func TestAnalyzeGeneratedFacePriorityMatrix(t *testing.T) {
	library := os.Getenv("ONNXRUNTIME_LIB")
	if library == "" {
		t.Fatal("integration tests require ONNXRUNTIME_LIB")
	}
	saliencyPath := os.Getenv("MODEL_PATH")
	if saliencyPath == "" {
		selected, err := configuredModelPath()
		if err != nil {
			t.Fatal(err)
		}
		saliencyPath = filepath.Join("..", "..", selected)
	}
	facePath := os.Getenv("FACE_MODEL_PATH")
	if facePath == "" {
		facePath = filepath.Join("..", "..", "models", "face_detection_yunet_2023mar.onnx")
	}

	saliencyModel, err := saliency.NewWithOptions(library, saliencyPath, saliency.Options{IntraOpThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := saliencyModel.Close(); err != nil {
			t.Error(err)
		}
	})
	faceModel, err := facedetection.NewWithOptions(library, facePath, facedetection.Options{IntraOpThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := faceModel.Close(); err != nil {
			t.Error(err)
		}
	})
	app := newApplicationWithFace(saliencyModel, faceModel, 2)

	cases := []facePriorityCase{
		{"generated-face-center.jpg", "face", 0.40, 0.60, 0.15, 0.40},
		{"generated-face-left.jpg", "face", 0.16, 0.38, 0.10, 0.40},
		{"generated-face-right.jpg", "face", 0.67, 0.88, 0.10, 0.42},
		{"generated-face-three-quarter.jpg", "face", 0.24, 0.54, 0.14, 0.48},
		{"generated-face-occluded.jpg", "face", 0.22, 0.74, 0.10, 0.52},
		{"generated-face-low-light.jpg", "face", 0.43, 0.76, 0.10, 0.50},
		{"generated-faces-primary.jpg", "face", 0.13, 0.44, 0.10, 0.48},
		{"generated-no-face-landscape.jpg", "saliency", 0, 0, 0, 0},
		{"generated-no-face-back-facing.jpg", "saliency", 0, 0, 0, 0},
		{"generated-no-face-blurred.jpg", "saliency", 0, 0, 0, 0},
	}

	analyze := func(t *testing.T, name string, data []byte) analyzeResponse {
		t.Helper()
		response := httptest.NewRecorder()
		app.handleAnalyze(response, fixtureRequest(t, testimages.Fixture{Name: name, ContentType: "image/jpeg"}, false, data))
		if response.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", response.Code, response.Body.String())
		}
		var result analyzeResponse
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		return result
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			data := testimages.Read(t, test.name)
			img, err := imageutil.Decode(data)
			if err != nil {
				t.Fatal(err)
			}
			var flipped bytes.Buffer
			if err := png.Encode(&flipped, imaging.FlipH(img)); err != nil {
				t.Fatal(err)
			}

			original := analyze(t, test.name, data)
			mirrored := analyze(t, "mirrored.png", flipped.Bytes())
			t.Logf("original=%+v mirrored=%+v", original, mirrored)
			for orientation, result := range map[string]analyzeResponse{
				"original": original,
				"mirrored": mirrored,
			} {
				if result.Source != test.wantSource {
					t.Errorf("%s source = %q, want %q", orientation, result.Source, test.wantSource)
				}
				if result.Confidence < 0 || result.Confidence > 1 {
					t.Errorf("%s confidence = %.4f", orientation, result.Confidence)
				}
			}
			if test.wantSource != "face" {
				return
			}
			if original.Gravity.X < test.minX || original.Gravity.X > test.maxX ||
				original.Gravity.Y < test.minY || original.Gravity.Y > test.maxY {
				t.Errorf("original primary face outside manually annotated region: %+v", original)
			}
			if mirrored.Gravity.X < 1-test.maxX || mirrored.Gravity.X > 1-test.minX ||
				mirrored.Gravity.Y < test.minY || mirrored.Gravity.Y > test.maxY {
				t.Errorf("mirrored primary face outside reflected annotation: %+v", mirrored)
			}
			if math.Abs(mirrored.Gravity.X-(1-original.Gravity.X)) > 0.04 ||
				math.Abs(mirrored.Gravity.Y-original.Gravity.Y) > 0.04 {
				t.Errorf("face selection is not reflection-consistent: original=%+v mirrored=%+v", original, mirrored)
			}
		})
	}
}

func TestConcurrentInferenceIsConsistent(t *testing.T) {
	library := os.Getenv("ONNXRUNTIME_LIB")
	if library == "" {
		t.Fatal("integration tests require ONNXRUNTIME_LIB")
	}
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		selected, err := configuredModelPath()
		if err != nil {
			t.Fatal(err)
		}
		modelPath = filepath.Join("..", "..", selected)
	}
	model, err := saliency.NewWithOptions(library, modelPath, saliency.Options{IntraOpThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := model.Close(); err != nil {
			t.Error(err)
		}
	})

	fixture := testimages.All[0]
	img, err := imageutil.Decode(testimages.Read(t, fixture.Name))
	if err != nil {
		t.Fatal(err)
	}
	input, _, err := imageutil.Prepare(img, saliency.InputWidth, saliency.InputHeight)
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.Infer(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	cancelCtx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := model.Infer(cancelCtx, input); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("cancelled inference error = %v, want context deadline exceeded", err)
	}

	const workers = 4
	var wait sync.WaitGroup
	errors := make(chan error, workers)
	for range workers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			got, err := model.Infer(context.Background(), input)
			if err != nil {
				errors <- err
				return
			}
			for i := range want {
				if got[i] != want[i] {
					errors <- fmt.Errorf("output[%d] = %v, want %v", i, got[i], want[i])
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
	// Infer returns independently owned Go memory, even after another call
	// uses different input and the native runtime has been destroyed.
	snapshot := append([]float32(nil), want...)
	if _, err := model.Infer(context.Background(), make([]float32, len(input))); err != nil {
		t.Fatal(err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if want[i] != snapshot[i] {
			t.Fatalf("returned output changed at %d after inference/Close", i)
		}
	}
}

func TestModelsShareRuntimeEnvironment(t *testing.T) {
	library := os.Getenv("ONNXRUNTIME_LIB")
	if library == "" {
		t.Fatal("integration tests require ONNXRUNTIME_LIB")
	}
	saliencyPath := filepath.Join("..", "..", "models", "u2net-int8.onnx")
	facePath := filepath.Join("..", "..", "models", "face_detection_yunet_2023mar.onnx")

	saliencyModel, err := saliency.NewWithOptions(library, saliencyPath, saliency.Options{IntraOpThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = saliencyModel.Close() })
	faceModel, err := facedetection.NewWithOptions(library, facePath, facedetection.Options{IntraOpThreads: 1})
	if err != nil {
		_ = saliencyModel.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = faceModel.Close() })

	img, err := imageutil.Decode(testimages.Read(t, "person-room.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	if detections, err := faceModel.Detect(context.Background(), img); err != nil || len(detections) == 0 {
		t.Fatalf("face detection = %+v, %v", detections, err)
	}
	if err := faceModel.Close(); err != nil {
		t.Fatal(err)
	}
	// Closing one session must not unload the process-wide runtime while the
	// saliency session still owns a reference.
	if _, err := saliencyModel.Infer(context.Background(), make([]float32, 3*saliency.InputWidth*saliency.InputHeight)); err != nil {
		t.Fatal(err)
	}
	if err := saliencyModel.Close(); err != nil {
		t.Fatal(err)
	}
}

// Missing prerequisites fail explicitly so CI cannot silently skip real inference.
func runSubjectCases(t *testing.T, cases []subjectCase) {
	t.Helper()
	library := os.Getenv("ONNXRUNTIME_LIB")
	if library == "" {
		t.Fatal("integration tests require ONNXRUNTIME_LIB")
	}
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		selected, err := configuredModelPath()
		if err != nil {
			t.Fatal(err)
		}
		modelPath = filepath.Join("..", "..", selected)
	}
	model, err := saliency.New(library, modelPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := model.Close(); err != nil {
			t.Error(err)
		}
	})
	app := newApplication(model, 2)
	analyze := func(t *testing.T, fixture testimages.Fixture, multi bool, data []byte) analyzeResponse {
		t.Helper()
		response := httptest.NewRecorder()
		app.handleAnalyze(response, fixtureRequest(t, fixture, multi, data))
		if response.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", response.Code, response.Body.String())
		}
		var result analyzeResponse
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(result.Gravity.X) || math.IsNaN(result.Gravity.Y) || result.Confidence < 0 || result.Confidence > 1 {
			t.Fatalf("unexpected output: %+v", result)
		}
		return result
	}

	for _, subject := range cases {
		fixture := testimages.Fixture{Name: subject.name, ContentType: "image/jpeg"}
		for _, format := range testimages.All {
			if format.Name == subject.name {
				fixture = format
				break
			}
		}
		t.Run(fixture.Name, func(t *testing.T) {
			data := testimages.Read(t, fixture.Name)
			raw := analyze(t, fixture, false, data)
			if raw.Confidence < 0.8 {
				t.Errorf("low confidence: %.4f", raw.Confidence)
			}
			t.Logf("gravity=(%.4f, %.4f), confidence=%.4f", raw.Gravity.X, raw.Gravity.Y, raw.Confidence)
			if raw.Gravity.X < subject.minX || raw.Gravity.X > subject.maxX || raw.Gravity.Y < subject.minY || raw.Gravity.Y > subject.maxY {
				t.Errorf("focal point outside manually annotated subject region: %+v", raw)
			}
			multipart := analyze(t, fixture, true, data)
			if math.Abs(raw.Gravity.X-multipart.Gravity.X) > 1e-5 || math.Abs(raw.Gravity.Y-multipart.Gravity.Y) > 1e-5 || math.Abs(raw.Confidence-multipart.Confidence) > 1e-5 {
				t.Fatalf("raw and multipart differ: %+v vs %+v", raw, multipart)
			}
			img, err := imageutil.Decode(data)
			if err != nil {
				t.Fatal(err)
			}
			var flipped bytes.Buffer
			if err := png.Encode(&flipped, imaging.FlipH(img)); err != nil {
				t.Fatal(err)
			}
			mirrored := analyze(t, testimages.Fixture{Name: "flipped.png", ContentType: "image/png"}, false, flipped.Bytes())
			if mirrored.Gravity.X < 1-subject.maxX || mirrored.Gravity.X > 1-subject.minX || mirrored.Gravity.Y < subject.minY || mirrored.Gravity.Y > subject.maxY {
				t.Errorf("mirrored focal point outside reflected subject region: %+v", mirrored)
			}
			// Saliency is not exactly reflection invariant. Allow a 7.5% displacement,
			// while still detecting fixed-center output or reversed coordinates.
			if math.Abs(mirrored.Gravity.X-(1-raw.Gravity.X)) > 0.075 || math.Abs(mirrored.Gravity.Y-raw.Gravity.Y) > 0.075 {
				t.Fatalf("mirror mismatch: original %+v, mirrored %+v", raw, mirrored)
			}
		})
	}
}
