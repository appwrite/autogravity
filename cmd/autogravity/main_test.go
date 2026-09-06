package main

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
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

type blockingReader struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

type gatedAnalyzer struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (a *gatedAnalyzer) Infer([]float32) ([]float32, error) {
	a.once.Do(func() { close(a.started) })
	<-a.release
	return make([]float32, saliency.InputWidth*saliency.InputHeight), nil
}

type countingReader struct {
	reader  *bytes.Reader
	started *atomic.Int32
	once    sync.Once
}

func (r *countingReader) Read(p []byte) (int, error) {
	r.once.Do(func() { r.started.Add(1) })
	return r.reader.Read(p)
}

func (r *blockingReader) Read([]byte) (int, error) {
	r.once.Do(func() { close(r.started) })
	<-r.release
	return 0, io.EOF
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

func TestHandleAnalyzeSlowUploadDoesNotTakeAnalysisSlot(t *testing.T) {
	app := newApplication(&fakeAnalyzer{})
	slowBody := &blockingReader{started: make(chan struct{}), release: make(chan struct{})}
	slowRequest := httptest.NewRequest(http.MethodPost, "/analyze", slowBody)
	slowRequest.Header.Set("Content-Type", "image/png")
	slowResponse := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		app.handleAnalyze(slowResponse, slowRequest)
	}()
	<-slowBody.started

	validRequest := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(testPNG(t)))
	validRequest.Header.Set("Content-Type", "image/png")
	validResponse := httptest.NewRecorder()
	app.handleAnalyze(validResponse, validRequest)
	if validResponse.Code != http.StatusOK {
		t.Fatalf("valid request status = %d, want 200", validResponse.Code)
	}

	close(slowBody.release)
	<-done
}

func TestHandleAnalyzeBoundsBufferedUploads(t *testing.T) {
	analyzer := &gatedAnalyzer{started: make(chan struct{}), release: make(chan struct{})}
	app := newApplication(analyzer)
	data := testPNG(t)

	var wait sync.WaitGroup
	startRequest := func(body io.Reader) {
		wait.Add(1)
		go func() {
			defer wait.Done()
			request := httptest.NewRequest(http.MethodPost, "/analyze", body)
			request.Header.Set("Content-Type", "image/png")
			response := httptest.NewRecorder()
			app.handleAnalyze(response, request)
			if response.Code != http.StatusOK {
				t.Errorf("status = %d, want 200", response.Code)
			}
		}()
	}

	startRequest(bytes.NewReader(data))
	<-analyzer.started // The analysis slot is occupied, but its upload slot is free.

	var bodiesStarted atomic.Int32
	for range maxConcurrentUploads + 1 {
		startRequest(&countingReader{reader: bytes.NewReader(data), started: &bodiesStarted})
	}
	deadline := time.After(time.Second)
	for bodiesStarted.Load() < maxConcurrentUploads {
		select {
		case <-deadline:
			t.Fatalf("only %d request bodies started", bodiesStarted.Load())
		default:
			time.Sleep(time.Millisecond)
		}
	}
	time.Sleep(20 * time.Millisecond)
	if got := bodiesStarted.Load(); got != maxConcurrentUploads {
		t.Fatalf("buffered request bodies = %d, want %d", got, maxConcurrentUploads)
	}

	close(analyzer.release)
	wait.Wait()
}

func TestHandleHealthz(t *testing.T) {
	app := newApplication(&fakeAnalyzer{})
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	response := httptest.NewRecorder()

	app.handleHealthz(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	if contentType := response.Header().Get("Content-Type"); contentType != "application/json" {
		t.Fatalf("Content-Type = %q, want %q", contentType, "application/json")
	}
	var body healthResponse
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Status != "ok" {
		t.Fatalf("status = %q, want %q", body.Status, "ok")
	}
}

func TestHandleHealthzErrors(t *testing.T) {
	t.Run("method not allowed", func(t *testing.T) {
		request := httptest.NewRequest(http.MethodPost, "/healthz", nil)
		response := httptest.NewRecorder()
		newApplication(&fakeAnalyzer{}).handleHealthz(response, request)
		if response.Code != http.StatusMethodNotAllowed {
			t.Fatalf("status = %d, want 405; body = %s", response.Code, response.Body.String())
		}
	})

	t.Run("model not ready", func(t *testing.T) {
		request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
		response := httptest.NewRecorder()
		newApplication(nil).handleHealthz(response, request)
		if response.Code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503; body = %s", response.Code, response.Body.String())
		}
	})
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
