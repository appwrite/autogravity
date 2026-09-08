package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"autogravity/internal/gravity"
	"autogravity/internal/imageutil"
	"autogravity/internal/saliency"
)

const (
	maxRequestBytes        = 10 << 20 // 10 MiB, including multipart overhead.
	maxConcurrentUploads   = 4        // Bounds buffered bodies without reserving inference.
	defaultIntraOpThreads  = 1        // Avoid N requests multiplying ONNX worker threads.
	defaultAnalysisTimeout = 30 * time.Second
	defaultShutdownTimeout = 30 * time.Second
)

var (
	version = "dev"
	commit  = "unknown"
)

type analyzer interface {
	// Infer borrows input only for the duration of the call. Returned data
	// must remain valid independently of input and subsequent inference calls.
	Infer(context.Context, []float32) ([]float32, error)
}

type application struct {
	model           analyzer
	uploadSlots     chan struct{}
	analysisSlots   chan struct{}
	inputBuffers    chan []float32
	telemetry       *telemetry
	analysisTimeout time.Duration
	draining        atomic.Bool
}

type analyzeResponse struct {
	Gravity    gravity.Point `json:"gravity"`
	Confidence float64       `json:"confidence"`
}

type healthResponse struct {
	Status string `json:"status"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	if err := run(); err != nil {
		slog.Error("service stopped", "error", err)
		os.Exit(1)
	}
}

func run() error {
	addr := envOrDefault("ADDR", ":8080")
	modelPath, err := configuredModelPath()
	if err != nil {
		return fmt.Errorf("startup: %w", err)
	}
	runtimePath := os.Getenv("ONNXRUNTIME_LIB")
	maxConcurrentAnalyses, err := positiveEnvInt("MAX_CONCURRENT_ANALYSES", runtime.GOMAXPROCS(0))
	if err != nil {
		return fmt.Errorf("startup: %w", err)
	}
	intraOpThreads, err := positiveEnvInt("ONNX_INTRA_OP_THREADS", defaultIntraOpThreads)
	if err != nil {
		return fmt.Errorf("startup: %w", err)
	}
	analysisTimeout, err := positiveEnvDuration("ANALYSIS_TIMEOUT", defaultAnalysisTimeout)
	if err != nil {
		return fmt.Errorf("startup: %w", err)
	}
	shutdownTimeout, err := positiveEnvDuration("SHUTDOWN_TIMEOUT", defaultShutdownTimeout)
	if err != nil {
		return fmt.Errorf("startup: %w", err)
	}

	model, err := saliency.NewWithOptions(runtimePath, modelPath, saliency.Options{
		IntraOpThreads: intraOpThreads,
	})
	if err != nil {
		return fmt.Errorf("startup: %w", err)
	}
	slog.Info("model loaded", "path", modelPath, "architecture", runtime.GOARCH,
		"analysis_slots", maxConcurrentAnalyses, "intra_op_threads", intraOpThreads)
	defer func() {
		if err := model.Close(); err != nil {
			slog.Error("failed to close model", "error", err)
		}
	}()

	app := newApplication(model, maxConcurrentAnalyses)
	app.analysisTimeout = analysisTimeout
	mux := http.NewServeMux()
	mux.HandleFunc("/analyze", app.handleAnalyze)
	mux.HandleFunc("/livez", app.handleLivez)
	mux.HandleFunc("/readyz", app.handleReadyz)
	mux.HandleFunc("/healthz", app.handleReadyz)
	mux.Handle("/metrics", app.telemetry.handler())

	server := &http.Server{
		Addr:              addr,
		Handler:           app.telemetry.instrument("", mux),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       analysisTimeout,
		WriteTimeout:      analysisTimeout + 5*time.Second,
		IdleTimeout:       60 * time.Second,
	}
	serviceCtx, cancelService := context.WithCancel(context.Background())
	defer cancelService()
	server.BaseContext = func(net.Listener) context.Context { return serviceCtx }

	shutdownSignals := make(chan os.Signal, 1)
	signal.Notify(shutdownSignals, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(shutdownSignals)
	serverErrors := make(chan error, 1)
	go func() {
		slog.Info("listening", "address", addr)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErrors <- err
		}
	}()

	select {
	case sig := <-shutdownSignals:
		slog.Info("shutting down", "signal", sig.String())
	case err := <-serverErrors:
		return fmt.Errorf("serve HTTP: %w", err)
	}

	app.draining.Store(true)
	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	if err := server.Shutdown(ctx); err != nil {
		cancel()
		cancelService() // Interrupt context-aware ONNX inference before force-closing sockets.
		if closeErr := server.Close(); closeErr != nil {
			return errors.Join(fmt.Errorf("graceful shutdown: %w", err), fmt.Errorf("force close: %w", closeErr))
		}
		return fmt.Errorf("graceful shutdown: %w", err)
	}
	cancel()
	return nil
}

// handleHealthz remains as a compatibility alias for callers and tests.
func (app *application) handleHealthz(w http.ResponseWriter, r *http.Request) {
	app.handleReadyz(w, r)
}

// An explicit path takes precedence. Otherwise precision is identical across
// architectures; invalid settings fail rather than silently changing quality.
func configuredModelPath() (string, error) {
	if path := os.Getenv("MODEL_PATH"); path != "" {
		return path, nil
	}
	switch precision := envOrDefault("MODEL_PRECISION", "int8"); precision {
	case "int8":
		return "models/u2net-int8.onnx", nil
	case "fp32":
		return "models/u2net.onnx", nil
	default:
		return "", fmt.Errorf("MODEL_PRECISION must be int8 or fp32, got %q", precision)
	}
}

func (app *application) handleLivez(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, healthResponse{Status: "ok"})
}

func (app *application) handleReadyz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if app.model == nil || app.draining.Load() {
		writeError(w, http.StatusServiceUnavailable, "model not ready")
		return
	}
	writeJSON(w, http.StatusOK, healthResponse{Status: "ok"})
}

func (app *application) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if app.draining.Load() {
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, "service is shutting down")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), app.analysisTimeout)
	defer cancel()
	r = r.WithContext(ctx)
	outcome := "success"
	defer func() { app.telemetry.analyses.WithLabelValues(outcome).Inc() }()
	observeCancellation := func() {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			outcome = "timeout"
			app.telemetry.cancellations.WithLabelValues("deadline").Inc()
		} else {
			outcome = "cancelled"
			app.telemetry.cancellations.WithLabelValues("client").Inc()
		}
	}
	uploadWaitStarted := time.Now()
	select {
	case app.uploadSlots <- struct{}{}:
		app.telemetry.stageDuration.WithLabelValues("upload_queue").Observe(time.Since(uploadWaitStarted).Seconds())
	case <-ctx.Done():
		observeCancellation()
		writeContextError(w, ctx)
		return
	}
	uploadSlotHeld := true
	defer func() {
		if uploadSlotHeld {
			<-app.uploadSlots
		}
	}()

	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
	defer r.Body.Close()
	data, err := readImage(r)
	if err != nil {
		if ctx.Err() != nil {
			observeCancellation()
			writeContextError(w, ctx)
			return
		}
		outcome = "invalid_request"
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeError(w, http.StatusRequestEntityTooLarge, "image exceeds the 10 MiB request limit")
			return
		}
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	analysisWaitStarted := time.Now()
	select {
	case app.analysisSlots <- struct{}{}:
		app.telemetry.stageDuration.WithLabelValues("analysis_queue").Observe(time.Since(analysisWaitStarted).Seconds())
		defer func() { <-app.analysisSlots }()
	case <-ctx.Done():
		observeCancellation()
		writeContextError(w, ctx)
		return
	}
	// Keep the upload slot while waiting for analysis so buffered request bodies
	// remain bounded. Release it once decoded-image admission is secured.
	<-app.uploadSlots
	uploadSlotHeld = false

	preprocessStarted := time.Now()
	img, err := imageutil.Decode(data)
	if err != nil {
		switch {
		case errors.Is(err, imageutil.ErrUnsupportedFormat):
			writeError(w, http.StatusUnsupportedMediaType, err.Error())
		case errors.Is(err, imageutil.ErrImageTooLarge):
			writeError(w, http.StatusRequestEntityTooLarge, err.Error())
		default:
			writeError(w, http.StatusBadRequest, "invalid image")
		}
		outcome = "invalid_image"
		return
	}

	var input []float32
	select {
	case input = <-app.inputBuffers:
	default:
		input = make([]float32, 3*saliency.InputWidth*saliency.InputHeight)
	}
	defer func() {
		select {
		case app.inputBuffers <- input:
		default:
		}
	}()
	content, err := imageutil.PrepareInto(input, img, saliency.InputWidth, saliency.InputHeight)
	if err != nil {
		outcome = "preprocessing_error"
		writeError(w, http.StatusInternalServerError, "failed to preprocess image")
		return
	}
	app.telemetry.stageDuration.WithLabelValues("preprocessing").Observe(time.Since(preprocessStarted).Seconds())
	inferenceStarted := time.Now()
	app.telemetry.inferenceActive.Inc()
	mapData, err := app.model.Infer(ctx, input)
	app.telemetry.inferenceActive.Dec()
	app.telemetry.stageDuration.WithLabelValues("inference").Observe(time.Since(inferenceStarted).Seconds())
	if err != nil {
		if ctx.Err() != nil {
			observeCancellation()
			writeContextError(w, ctx)
			return
		}
		outcome = "inference_error"
		slog.Error("inference failed", "error", err)
		writeError(w, http.StatusInternalServerError, "image analysis failed")
		return
	}
	if ctx.Err() != nil {
		observeCancellation()
		writeContextError(w, ctx)
		return
	}
	focalStarted := time.Now()
	point, confidence, err := gravity.FromSaliencyRegion(
		mapData,
		saliency.InputWidth,
		saliency.InputHeight,
		content,
	)
	if err != nil {
		outcome = "focal_point_error"
		slog.Error("focal-point calculation failed", "error", err)
		writeError(w, http.StatusInternalServerError, "image analysis failed")
		return
	}
	app.telemetry.stageDuration.WithLabelValues("focal_point").Observe(time.Since(focalStarted).Seconds())
	if ctx.Err() != nil {
		observeCancellation()
		writeContextError(w, ctx)
		return
	}

	writeJSON(w, http.StatusOK, analyzeResponse{Gravity: point, Confidence: confidence})
}

func newApplication(model analyzer, maxConcurrentAnalyses int) *application {
	return &application{
		model:           model,
		uploadSlots:     make(chan struct{}, maxConcurrentUploads),
		analysisSlots:   make(chan struct{}, maxConcurrentAnalyses),
		inputBuffers:    make(chan []float32, maxConcurrentAnalyses),
		telemetry:       newTelemetry(envOrDefault("MODEL_PRECISION", "int8"), maxConcurrentAnalyses),
		analysisTimeout: defaultAnalysisTimeout,
	}
}

func writeContextError(w http.ResponseWriter, ctx context.Context) {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		writeError(w, http.StatusGatewayTimeout, "analysis timed out")
		return
	}
	writeError(w, 499, "request cancelled")
}

func readImage(r *http.Request) ([]byte, error) {
	contentType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil {
		return nil, errors.New("invalid Content-Type header")
	}
	if contentType != "multipart/form-data" {
		data, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, fmt.Errorf("read image: %w", err)
		}
		if len(data) == 0 {
			return nil, errors.New("empty image body")
		}
		return data, nil
	}

	boundary := params["boundary"]
	if boundary == "" {
		return nil, errors.New("multipart boundary is missing")
	}
	reader, err := r.MultipartReader()
	if err != nil {
		return nil, fmt.Errorf("read multipart body: %w", err)
	}
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read multipart body: %w", err)
		}
		if part.FormName() != "image" {
			part.Close()
			continue
		}
		data, readErr := io.ReadAll(part)
		part.Close()
		if readErr != nil {
			return nil, fmt.Errorf("read image field: %w", readErr)
		}
		if len(data) == 0 {
			return nil, errors.New("image field is empty")
		}
		return data, nil
	}
	return nil, errors.New("multipart field \"image\" is required")
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorResponse{Error: message})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		slog.Error("write response failed", "error", err)
	}
}

func envOrDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func positiveEnvInt(key string, fallback int) (int, error) {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 1 {
		return 0, fmt.Errorf("%s must be a positive integer, got %q", key, value)
	}
	return parsed, nil
}

func positiveEnvDuration(key string, fallback time.Duration) (time.Duration, error) {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback, nil
	}
	parsed, err := time.ParseDuration(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s must be a positive duration, got %q", key, value)
	}
	return parsed, nil
}
