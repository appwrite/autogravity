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

// This suite deliberately fails when prerequisites are absent: CI must not
// silently skip the only tests that exercise the real model.
func TestAnalyzeRealModel(t *testing.T) {
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
		if math.IsNaN(result.Gravity.X) || math.IsNaN(result.Gravity.Y) || result.Confidence < 0.8 || result.Confidence > 1 {
			t.Fatalf("unexpected output: %+v", result)
		}
		return result
	}
	for _, fixture := range testimages.All {
		t.Run(fixture.Name, func(t *testing.T) {
			data := testimages.Read(t, fixture.Name)
			raw := analyze(t, fixture, false, data)
			t.Logf("gravity=(%.4f, %.4f), confidence=%.4f", raw.Gravity.X, raw.Gravity.Y, raw.Confidence)
			// The flower occupies the left/center; the plush toy occupies the center.
			minX, maxX, minY, maxY := 0.25, 0.47, 0.30, 0.70
			if fixture.Name == "portrait.jpg" {
				minX, maxX, minY, maxY = 0.35, 0.65, 0.30, 0.70
			}
			if raw.Gravity.X < minX || raw.Gravity.X > maxX || raw.Gravity.Y < minY || raw.Gravity.Y > maxY {
				t.Fatalf("focal point outside subject region: %+v", raw)
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
			// Saliency is not exactly reflection invariant. Allow a 5% displacement,
			// while still detecting fixed-center output or reversed coordinates.
			if math.Abs(mirrored.Gravity.X-(1-raw.Gravity.X)) > 0.05 || math.Abs(mirrored.Gravity.Y-raw.Gravity.Y) > 0.05 {
				t.Fatalf("mirror mismatch: original %+v, mirrored %+v", raw, mirrored)
			}
		})
	}
}
