//go:build integration

package main

import (
	"bytes"
	"encoding/json"
	"image/png"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

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
		{"bird-wire.jpg", 0.44, 0.53, 0.44, 0.61},
	})
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
		modelPath = filepath.Join("..", "..", "models", "u2netp.onnx")
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
	app := newApplication(model)
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
