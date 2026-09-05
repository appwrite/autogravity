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

const maxRequestBytes = 10 << 20 // 10 MiB, including multipart overhead.

type analyzer interface {
	Infer([]float32) ([]float32, error)
}

type application struct {
	model analyzer
}

type analyzeResponse struct {
	Gravity    gravity.Point `json:"gravity"`
	Confidence float64       `json:"confidence"`
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
	modelPath := envOrDefault("MODEL_PATH", "models/u2netp.onnx")
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

	app := &application{model: model}
	mux := http.NewServeMux()
	mux.HandleFunc("/analyze", app.handleAnalyze)

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

func (app *application) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

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

	input, err := imageutil.Prepare(img, saliency.InputWidth, saliency.InputHeight)
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
	point, confidence, err := gravity.FromSaliency(mapData, saliency.InputWidth, saliency.InputHeight)
	if err != nil {
		slog.Error("focal-point calculation failed", "error", err)
		writeError(w, http.StatusInternalServerError, "image analysis failed")
		return
	}

	writeJSON(w, http.StatusOK, analyzeResponse{Gravity: point, Confidence: confidence})
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
