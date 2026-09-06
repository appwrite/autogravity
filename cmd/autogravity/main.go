package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"autogravity/internal/gravity"
	"autogravity/internal/imageutil"
	"autogravity/internal/saliency"
)

const (
	maxRequestBytes       = 10 << 20 // 10 MiB, including multipart overhead.
	maxConcurrentUploads  = 4        // Bounds buffered bodies without reserving inference.
	maxConcurrentAnalyses = 1        // Inference is serialized; bound decoded-image memory too.
)

type analyzer interface {
	Infer([]float32) ([]float32, error)
}

type application struct {
	model         analyzer
	uploadSlots   chan struct{}
	analysisSlots chan struct{}
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
	if err := run(); err != nil {
		slog.Error("service stopped", "error", err)
		os.Exit(1)
	}
}

func run() error {
	addr := envOrDefault("ADDR", ":8080")
	modelPath := envOrDefault("MODEL_PATH", "models/u2net.onnx")
	runtimePath := os.Getenv("ONNXRUNTIME_LIB")

	model, err := saliency.New(runtimePath, modelPath)
	if err != nil {
		return fmt.Errorf("startup: %w", err)
	}
	defer func() {
		if err := model.Close(); err != nil {
			slog.Error("failed to close model", "error", err)
		}
	}()

	app := newApplication(model)
	mux := http.NewServeMux()
	mux.HandleFunc("/analyze", app.handleAnalyze)
	mux.HandleFunc("/healthz", app.handleHealthz)

	server := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		return fmt.Errorf("graceful shutdown: %w", err)
	}
	return nil
}

func (app *application) handleHealthz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if app.model == nil {
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
	select {
	case app.uploadSlots <- struct{}{}:
	case <-r.Context().Done():
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
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeError(w, http.StatusRequestEntityTooLarge, "image exceeds the 10 MiB request limit")
			return
		}
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	select {
	case app.analysisSlots <- struct{}{}:
		defer func() { <-app.analysisSlots }()
	case <-r.Context().Done():
		return
	}
	// Keep the upload slot while waiting for analysis so buffered request bodies
	// remain bounded. Release it once decoded-image admission is secured.
	<-app.uploadSlots
	uploadSlotHeld = false

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
		return
	}

	input, content, err := imageutil.Prepare(img, saliency.InputWidth, saliency.InputHeight)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to preprocess image")
		return
	}
	mapData, err := app.model.Infer(input)
	if err != nil {
		slog.Error("inference failed", "error", err)
		writeError(w, http.StatusInternalServerError, "image analysis failed")
		return
	}
	point, confidence, err := gravity.FromSaliencyRegion(
		mapData,
		saliency.InputWidth,
		saliency.InputHeight,
		content,
	)
	if err != nil {
		slog.Error("focal-point calculation failed", "error", err)
		writeError(w, http.StatusInternalServerError, "image analysis failed")
		return
	}

	writeJSON(w, http.StatusOK, analyzeResponse{Gravity: point, Confidence: confidence})
}

func newApplication(model analyzer) *application {
	return &application{
		model:         model,
		uploadSlots:   make(chan struct{}, maxConcurrentUploads),
		analysisSlots: make(chan struct{}, maxConcurrentAnalyses),
	}
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
