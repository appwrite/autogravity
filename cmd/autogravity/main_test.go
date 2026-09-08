package main

import (
	"bytes"
	"context"
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

type contextAnalyzer struct{}

func (contextAnalyzer) Infer(ctx context.Context, _ []float32) ([]float32, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

type cancelAfterAnalyzer struct{ cancel context.CancelFunc }

func (a cancelAfterAnalyzer) Infer(_ context.Context, _ []float32) ([]float32, error) {
	result := make([]float32, saliency.InputWidth*saliency.InputHeight)
	result[len(result)/2] = 0.9
	a.cancel()
	return result, nil
}

func TestConfiguredModelPath(t *testing.T) {
	for _, tc := range []struct {
		name, precision, path, want string
		invalid                     bool
	}{
		{"default", "", "", "models/u2net-int8.onnx", false},
		{"int8", "int8", "", "models/u2net-int8.onnx", false},
		{"fp32", "fp32", "", "models/u2net.onnx", false},
		{"invalid", "fp16", "", "", true},
		{"override", "int8", "/custom/fp32.onnx", "/custom/fp32.onnx", false},
		{"override invalid precision", "fp16", "/custom/model.onnx", "/custom/model.onnx", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("MODEL_PATH", tc.path)
			t.Setenv("MODEL_PRECISION", tc.precision)
			got, err := configuredModelPath()
			if (err != nil) != tc.invalid {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
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

func (a *gatedAnalyzer) Infer(context.Context, []float32) ([]float32, error) {
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

func (f *fakeAnalyzer) Infer(_ context.Context, input []float32) ([]float32, error) {
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
	app := newApplication(&fakeAnalyzer{}, 2)
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
	newApplication(&fakeAnalyzer{}, 2).handleAnalyze(response, request)

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
			newApplication(&fakeAnalyzer{}, 2).handleAnalyze(response, request)
			if response.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d; body = %s", response.Code, tt.wantStatus, response.Body.String())
			}
		})
	}
}

func TestHandleAnalyzeBoundsConcurrentWork(t *testing.T) {
	analyzer := &fakeAnalyzer{delay: 10 * time.Millisecond}
	const concurrency = 2
	app := newApplication(analyzer, concurrency)
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

	if got := analyzer.maxInFlight.Load(); got != concurrency {
		t.Fatalf("maximum concurrent analyses = %d, want %d", got, concurrency)
	}
}

func TestPositiveEnvInt(t *testing.T) {
	const key = "TEST_CONCURRENCY"
	for _, tt := range []struct {
		name  string
		value string
		want  int
		bad   bool
	}{
		{name: "unset", want: 2},
		{name: "valid", value: "4", want: 4},
		{name: "trimmed", value: " 3 ", want: 3},
		{name: "zero", value: "0", bad: true},
		{name: "negative", value: "-1", bad: true},
		{name: "not a number", value: "many", bad: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(key, tt.value)
			got, err := positiveEnvInt(key, 2)
			if (err != nil) != tt.bad || got != tt.want {
				t.Fatalf("positiveEnvInt() = %d, %v; want %d, bad=%v", got, err, tt.want, tt.bad)
			}
		})
	}
}

func TestPositiveEnvDuration(t *testing.T) {
	const key = "TEST_DURATION"
	for _, tt := range []struct {
		value string
		want  time.Duration
		bad   bool
	}{
		{"", time.Second, false},
		{"250ms", 250 * time.Millisecond, false},
		{"0s", 0, true},
		{"later", 0, true},
	} {
		t.Setenv(key, tt.value)
		got, err := positiveEnvDuration(key, time.Second)
		if (err != nil) != tt.bad || got != tt.want {
			t.Fatalf("positiveEnvDuration(%q) = %v, %v; want %v, bad=%v", tt.value, got, err, tt.want, tt.bad)
		}
	}
}

func TestHandleAnalyzeDeadlineCancelsInference(t *testing.T) {
	app := newApplication(contextAnalyzer{}, 1)
	app.analysisTimeout = 10 * time.Millisecond
	request := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(testPNG(t)))
	request.Header.Set("Content-Type", "image/png")
	response := httptest.NewRecorder()

	app.handleAnalyze(response, request)

	if response.Code != http.StatusGatewayTimeout {
		t.Fatalf("status = %d, want 504; body = %s", response.Code, response.Body.String())
	}
	if len(app.uploadSlots) != 0 || len(app.analysisSlots) != 0 {
		t.Fatal("admission slots leaked after cancellation")
	}
}

func TestHandleAnalyzeChecksCancellationAfterInference(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	app := newApplication(cancelAfterAnalyzer{cancel: cancel}, 1)
	request := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(testPNG(t))).WithContext(ctx)
	request.Header.Set("Content-Type", "image/png")
	response := httptest.NewRecorder()

	app.handleAnalyze(response, request)

	if response.Code != 499 {
		t.Fatalf("status = %d, want 499; body = %s", response.Code, response.Body.String())
	}
}

func TestDrainingRejectsAnalysisAndReadiness(t *testing.T) {
	app := newApplication(&fakeAnalyzer{}, 1)
	app.draining.Store(true)

	analyze := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/analyze", bytes.NewReader(testPNG(t)))
	request.Header.Set("Content-Type", "image/png")
	app.handleAnalyze(analyze, request)
	if analyze.Code != http.StatusServiceUnavailable || analyze.Header().Get("Retry-After") != "1" {
		t.Fatalf("analyze response = %d, Retry-After %q", analyze.Code, analyze.Header().Get("Retry-After"))
	}

	ready := httptest.NewRecorder()
	app.handleReadyz(ready, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if ready.Code != http.StatusServiceUnavailable {
		t.Fatalf("ready status = %d, want 503", ready.Code)
	}

	live := httptest.NewRecorder()
	app.handleLivez(live, httptest.NewRequest(http.MethodGet, "/livez", nil))
	if live.Code != http.StatusOK {
		t.Fatalf("live status = %d, want 200", live.Code)
	}
}

func TestTelemetryPropagatesSafeRequestID(t *testing.T) {
	tel := newTelemetry("int8", 1)
	handler := tel.instrument("/livez", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	request := httptest.NewRequest(http.MethodGet, "/livez", nil)
	request.Header.Set("X-Request-ID", "request_123")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if got := response.Header().Get("X-Request-ID"); got != "request_123" {
		t.Fatalf("request ID = %q", got)
	}
}

func TestTelemetryCoversUnmatchedRoutes(t *testing.T) {
	tel := newTelemetry("int8", 1)
	handler := tel.instrument("", http.NewServeMux())
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/missing", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", response.Code)
	}
	if response.Header().Get("X-Request-ID") == "" {
		t.Fatal("unmatched response has no request ID")
	}
}

func TestHandleAnalyzeSlowUploadDoesNotTakeAnalysisSlot(t *testing.T) {
	app := newApplication(&fakeAnalyzer{}, 2)
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
	// This test isolates upload admission by deliberately occupying the only
	// analysis slot. Concurrent analysis is covered separately above.
	app := newApplication(analyzer, 1)
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
	app := newApplication(&fakeAnalyzer{}, 2)
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
		newApplication(&fakeAnalyzer{}, 2).handleHealthz(response, request)
		if response.Code != http.StatusMethodNotAllowed {
			t.Fatalf("status = %d, want 405; body = %s", response.Code, response.Body.String())
		}
	})

	t.Run("model not ready", func(t *testing.T) {
		request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
		response := httptest.NewRecorder()
		newApplication(nil, 2).handleHealthz(response, request)
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
