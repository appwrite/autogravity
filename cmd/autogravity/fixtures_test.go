package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"autogravity/internal/saliency"
	"autogravity/internal/testimages"
)

type analyzerFunc func([]float32) ([]float32, error)

func (f analyzerFunc) Infer(_ context.Context, input []float32) ([]float32, error) { return f(input) }

func fixtureRequest(t *testing.T, fixture testimages.Fixture, multipartBody bool, data []byte) *http.Request {
	t.Helper()
	var body io.Reader = bytes.NewReader(data)
	contentType := fixture.ContentType
	if multipartBody {
		var encoded bytes.Buffer
		writer := multipart.NewWriter(&encoded)
		if err := writer.WriteField("description", "ignored metadata before the image"); err != nil {
			t.Fatal(err)
		}
		part, err := writer.CreateFormFile("image", fixture.Name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = part.Write(data); err != nil {
			t.Fatal(err)
		}
		if err = writer.Close(); err != nil {
			t.Fatal(err)
		}
		body = &encoded
		contentType = writer.FormDataContentType()
	}
	request := httptest.NewRequest(http.MethodPost, "/analyze", body)
	request.Header.Set("Content-Type", contentType)
	return request
}

func TestHandleAnalyzeFixtures(t *testing.T) {
	for _, fixture := range testimages.All {
		for _, multipartBody := range []bool{false, true} {
			name := fixture.Name + "/raw"
			if multipartBody {
				name = fixture.Name + "/multipart"
			}
			t.Run(name, func(t *testing.T) {
				calls := 0
				model := analyzerFunc(func(input []float32) ([]float32, error) {
					calls++
					if len(input) != 3*saliency.InputWidth*saliency.InputHeight {
						t.Fatalf("input length = %d", len(input))
					}
					result := make([]float32, saliency.InputWidth*saliency.InputHeight)
					// A padding activation must affect neither the point nor confidence.
					result[0] = 1
					x, y := fixture.Content.Max.X-1, fixture.Content.Min.Y
					result[y*saliency.InputWidth+x] = 0.75
					return result, nil
				})
				app := newApplication(model, 2)
				response := httptest.NewRecorder()
				app.handleAnalyze(response, fixtureRequest(t, fixture, multipartBody, testimages.Read(t, fixture.Name)))
				if response.Code != http.StatusOK {
					t.Fatalf("status = %d: %s", response.Code, response.Body.String())
				}
				if got := response.Header().Get("Content-Type"); got != "application/json" {
					t.Fatalf("Content-Type = %q", got)
				}
				var body analyzeResponse
				if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
					t.Fatal(err)
				}
				if body.Gravity.X != 1 || body.Gravity.Y != 0 || math.Abs(body.Confidence-0.75) > 1e-6 {
					t.Fatalf("response = %+v, want top-right with confidence 0.75", body)
				}
				if calls != 1 {
					t.Fatalf("inference calls = %d", calls)
				}
				if len(app.uploadSlots) != 0 || len(app.analysisSlots) != 0 {
					t.Fatal("admission slots leaked")
				}
			})
		}
	}
}

func TestHandleAnalyzeTruncatedFixtures(t *testing.T) {
	for _, fixture := range testimages.All {
		for _, multipartBody := range []bool{false, true} {
			name := fixture.Name + "/raw"
			if multipartBody {
				name = fixture.Name + "/multipart"
			}
			t.Run(name, func(t *testing.T) {
				model := analyzerFunc(func([]float32) ([]float32, error) { t.Error("inference called for corrupt image"); return nil, nil })
				app := newApplication(model, 2)
				data := testimages.Read(t, fixture.Name)
				response := httptest.NewRecorder()
				app.handleAnalyze(response, fixtureRequest(t, fixture, multipartBody, data[:len(data)/2]))
				if response.Code != http.StatusBadRequest {
					t.Fatalf("status = %d: %s", response.Code, response.Body.String())
				}
				var body errorResponse
				if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
					t.Fatal(err)
				}
				if body.Error != "invalid image" {
					t.Fatalf("error = %q", body.Error)
				}
				if len(app.uploadSlots) != 0 || len(app.analysisSlots) != 0 {
					t.Fatal("admission slots leaked")
				}
			})
		}
	}
}

func TestHandleAnalyzeInferenceFailureRecovery(t *testing.T) {
	for _, invalidMap := range []bool{false, true} {
		name := "inference error"
		if invalidMap {
			name = "invalid output dimensions"
		}
		t.Run(name, func(t *testing.T) {
			calls := 0
			app := newApplication(analyzerFunc(func([]float32) ([]float32, error) {
				calls++
				if calls == 1 {
					if invalidMap {
						return []float32{1}, nil
					}
					return nil, errors.New("private runtime error")
				}
				return make([]float32, 320*320), nil
			}), 2)
			fixture := testimages.All[0]
			for _, status := range []int{http.StatusInternalServerError, http.StatusOK} {
				response := httptest.NewRecorder()
				app.handleAnalyze(response, fixtureRequest(t, fixture, false, testimages.Read(t, fixture.Name)))
				if response.Code != status {
					t.Fatalf("status = %d, want %d: %s", response.Code, status, response.Body.String())
				}
				if bytes.Contains(response.Body.Bytes(), []byte("private runtime")) {
					t.Fatal("runtime details exposed")
				}
				if len(app.uploadSlots) != 0 || len(app.analysisSlots) != 0 {
					t.Fatal("admission slots leaked")
				}
			}
		})
	}
}
