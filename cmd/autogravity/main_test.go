package main

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"autogravity/internal/saliency"
)

type fakeAnalyzer struct {
	delay       time.Duration
	inFlight    atomic.Int32
	maxInFlight atomic.Int32
}

func (f *fakeAnalyzer) Infer(input []float32) ([]float32, error) {
	current := f.inFlight.Add(1)
	defer f.inFlight.Add(-1)
	for {
		maximum := f.maxInFlight.Load()
		if current <= maximum || f.maxInFlight.CompareAndSwap(maximum, current) {
			break
		}
	}
	time.Sleep(f.delay)
	result := make([]float32, saliency.InputWidth*saliency.InputHeight)
	result[len(result)/2] = 0.9
	return result, nil
}

func TestHandleAnalyzeRawImage(t *testing.T) {
	app := newApplication(&fakeAnalyzer{})
	request := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(testPNG(t)))
	request.Header.Set("Content-Type", "image/png")
	response := httptest.NewRecorder()

	app.handleAnalyze(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	var body analyzeResponse
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Gravity.X < 0 || body.Gravity.X > 1 || body.Gravity.Y < 0 || body.Gravity.Y > 1 {
		t.Fatalf("gravity is not normalized: %+v", body.Gravity)
	}
	if body.Confidence < 0 || body.Confidence > 1 {
		t.Fatalf("confidence is not normalized: %v", body.Confidence)
	}
}

func TestHandleAnalyzeMultipartImage(t *testing.T) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("image", "test.png")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(testPNG(t)); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, "/analyze", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()
	newApplication(&fakeAnalyzer{}).handleAnalyze(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", response.Code, response.Body.String())
	}
}

func TestHandleAnalyzeErrors(t *testing.T) {
	tests := []struct {
		name        string
		method      string
		contentType string
		body        []byte
		wantStatus  int
	}{
		{name: "method", method: http.MethodGet, wantStatus: http.StatusMethodNotAllowed},
		{name: "unsupported format", method: http.MethodPost, contentType: "application/octet-stream", body: []byte("not an image"), wantStatus: http.StatusUnsupportedMediaType},
		{name: "empty body", method: http.MethodPost, contentType: "image/png", wantStatus: http.StatusBadRequest},
		{name: "request too large", method: http.MethodPost, contentType: "image/png", body: make([]byte, maxRequestBytes+1), wantStatus: http.StatusRequestEntityTooLarge},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := httptest.NewRequest(tt.method, "/analyze", bytes.NewReader(tt.body))
			if tt.contentType != "" {
				request.Header.Set("Content-Type", tt.contentType)
			}
			response := httptest.NewRecorder()
			newApplication(&fakeAnalyzer{}).handleAnalyze(response, request)
			if response.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d; body = %s", response.Code, tt.wantStatus, response.Body.String())
			}
		})
	}
}

func TestHandleAnalyzeBoundsConcurrentWork(t *testing.T) {
	analyzer := &fakeAnalyzer{delay: 10 * time.Millisecond}
	app := newApplication(analyzer)
	data := testPNG(t)

	var wait sync.WaitGroup
	for range 8 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			request := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(data))
			request.Header.Set("Content-Type", "image/png")
			response := httptest.NewRecorder()
			app.handleAnalyze(response, request)
			if response.Code != http.StatusOK {
				t.Errorf("status = %d, want 200", response.Code)
			}
		}()
	}
	wait.Wait()

	if got := analyzer.maxInFlight.Load(); got != maxConcurrentAnalyses {
		t.Fatalf("maximum concurrent analyses = %d, want %d", got, maxConcurrentAnalyses)
	}
}

func testPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.SetRGBA(0, 0, color.RGBA{R: 255, A: 255})
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, img); err != nil {
		t.Fatal(err)
	}
	return encoded.Bytes()
}
