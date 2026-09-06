package benchmark_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"autogravity/internal/gravity"
	"autogravity/internal/imageutil"
	"autogravity/internal/saliency"
)

var (
	benchmarkPoint      gravity.Point
	benchmarkConfidence float64
)

// BenchmarkAnalyze measures the complete per-request analysis pipeline. Model
// initialization is deliberately excluded because production loads it once.
func BenchmarkAnalyze(b *testing.B) {
	runtimeLibrary := os.Getenv("ONNXRUNTIME_LIB")
	if runtimeLibrary == "" {
		b.Skip("set ONNXRUNTIME_LIB to run the inference benchmark")
	}

	_, thisFile, _, _ := runtime.Caller(0)
	projectRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = filepath.Join(projectRoot, "models", "u2netp.onnx")
	}

	model, err := saliency.New(runtimeLibrary, modelPath)
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	b.Cleanup(func() {
		if err := model.Close(); err != nil {
			b.Errorf("close model: %v", err)
		}
	})

	fixtures := []string{"landscape.jpg", "portrait.png"}
	for _, fixture := range fixtures {
		data, err := os.ReadFile(filepath.Join(projectRoot, "internal", "benchmark", "testdata", fixture))
		if err != nil {
			b.Fatalf("read %s: %v", fixture, err)
		}

		b.Run(fixture, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			for range b.N {
				img, err := imageutil.Decode(data)
				if err != nil {
					b.Fatal(err)
				}
				input, content, err := imageutil.Prepare(img, saliency.InputWidth, saliency.InputHeight)
				if err != nil {
					b.Fatal(err)
				}
				mapData, err := model.Infer(input)
				if err != nil {
					b.Fatal(err)
				}
				benchmarkPoint, benchmarkConfidence, err = gravity.FromSaliencyRegion(
					mapData,
					saliency.InputWidth,
					saliency.InputHeight,
					content,
				)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
