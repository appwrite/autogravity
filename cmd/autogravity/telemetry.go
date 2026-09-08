package main

import (
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type telemetry struct {
	registry        *prometheus.Registry
	httpRequests    *prometheus.CounterVec
	httpDuration    *prometheus.HistogramVec
	httpActive      prometheus.Gauge
	stageDuration   *prometheus.HistogramVec
	analyses        *prometheus.CounterVec
	inferenceActive prometheus.Gauge
	cancellations   *prometheus.CounterVec
}

func newTelemetry(modelPrecision string, maxConcurrentAnalyses int) *telemetry {
	registry := prometheus.NewRegistry()
	t := &telemetry{
		registry: registry,
		httpRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "autogravity", Subsystem: "http", Name: "requests_total",
			Help: "HTTP requests by route, method, and status code.",
		}, []string{"route", "method", "status"}),
		httpDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "autogravity", Subsystem: "http", Name: "request_duration_seconds",
			Help: "HTTP request duration by route and method.",
		}, []string{"route", "method"}),
		httpActive: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "autogravity", Subsystem: "http", Name: "requests_active",
			Help: "HTTP requests currently being served.",
		}),
		stageDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "autogravity", Name: "stage_duration_seconds",
			Help: "Analysis pipeline stage duration.",
		}, []string{"stage"}),
		analyses: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "autogravity", Name: "analyses_total",
			Help: "Image analyses by outcome.",
		}, []string{"outcome"}),
		inferenceActive: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "autogravity", Name: "inference_active",
			Help: "ONNX inference calls currently running.",
		}),
		cancellations: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "autogravity", Name: "cancellations_total",
			Help: "Cancelled analyses by reason.",
		}, []string{"reason"}),
	}
	registry.MustRegister(t.httpRequests, t.httpDuration, t.httpActive, t.stageDuration,
		t.analyses, t.inferenceActive, t.cancellations, prometheus.NewGoCollector(),
		prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	registry.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "autogravity", Name: "build_info", Help: "Build and model configuration.",
		ConstLabels: prometheus.Labels{
			"version": version, "commit": commit, "model_precision": modelPrecision,
			"max_concurrent_analyses": strconv.Itoa(maxConcurrentAnalyses),
		},
	}, func() float64 { return 1 }))
	return t
}

func (t *telemetry) handler() http.Handler {
	return promhttp.HandlerFor(t.registry, promhttp.HandlerOpts{})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (w *statusRecorder) WriteHeader(status int) {
	if w.status != 0 {
		return
	}
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusRecorder) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	n, err := w.ResponseWriter.Write(data)
	w.bytes += n
	return n, err
}

func (t *telemetry) instrument(route string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		requestID := validRequestID(r.Header.Get("X-Request-ID"))
		if requestID == "" {
			requestID = newRequestID()
		}
		w.Header().Set("X-Request-ID", requestID)
		recorder := &statusRecorder{ResponseWriter: w}
		t.httpActive.Inc()
		defer func() {
			t.httpActive.Dec()
			status := recorder.status
			if status == 0 {
				status = http.StatusOK
			}
			duration := time.Since(started)
			t.httpRequests.WithLabelValues(route, r.Method, strconv.Itoa(status)).Inc()
			t.httpDuration.WithLabelValues(route, r.Method).Observe(duration.Seconds())
			slog.Info("request completed", "request_id", requestID, "method", r.Method,
				"route", route, "status", status, "response_bytes", recorder.bytes,
				"duration_ms", duration.Milliseconds())
		}()
		next.ServeHTTP(recorder, r)
	})
}

func validRequestID(value string) string {
	if len(value) < 1 || len(value) > 64 {
		return ""
	}
	for _, char := range value {
		if !(char == '-' || char == '_' || char >= '0' && char <= '9' ||
			char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z') {
			return ""
		}
	}
	return value
}

func newRequestID() string {
	var data [16]byte
	if _, err := rand.Read(data[:]); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return hex.EncodeToString(data[:])
}
