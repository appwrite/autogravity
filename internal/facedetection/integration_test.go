//go:build integration

package facedetection

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"autogravity/internal/imageutil"
	"autogravity/internal/testimages"
	"github.com/disintegration/imaging"
)

func TestDetectRealModel(t *testing.T) {
	library := os.Getenv("ONNXRUNTIME_LIB")
	if library == "" {
		t.Fatal("integration tests require ONNXRUNTIME_LIB")
	}
	modelPath := os.Getenv("FACE_MODEL_PATH")
	if modelPath == "" {
		modelPath = filepath.Join("..", "..", "models", "face_detection_yunet_2023mar.onnx")
	}
	model, err := NewWithOptions(library, modelPath, Options{IntraOpThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := model.Close(); err != nil {
			t.Error(err)
		}
	})

	img, err := imageutil.Decode(testimages.Read(t, "person-room.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	detections, err := model.Detect(context.Background(), img)
	if err != nil {
		t.Fatal(err)
	}
	face, ok := Primary(detections)
	if !ok {
		t.Fatal("clear portrait produced no face detection")
	}
	x, y := face.Center()
	if x < 0.43 || x > 0.58 || y < 0.40 || y > 0.62 {
		t.Fatalf("face center = (%.4f, %.4f), outside annotated region", x, y)
	}
	if face.Confidence < DefaultScoreThreshold || face.Confidence > 1 {
		t.Fatalf("face confidence = %.4f", face.Confidence)
	}

	mirroredDetections, err := model.Detect(context.Background(), imaging.FlipH(img))
	if err != nil {
		t.Fatal(err)
	}
	mirrored, ok := Primary(mirroredDetections)
	if !ok {
		t.Fatal("mirrored portrait produced no face detection")
	}
	mirroredX, mirroredY := mirrored.Center()
	if math.Abs(mirroredX-(1-x)) > 0.03 || math.Abs(mirroredY-y) > 0.03 {
		t.Fatalf("mirrored face center = (%.4f, %.4f), original = (%.4f, %.4f)", mirroredX, mirroredY, x, y)
	}

	for _, name := range []string{
		"rose.png", "portrait.jpg", "dog-portrait.jpg", "puppies.jpg",
		"panda-bamboo.jpg", "pedestrian-dog.jpg", "bird-branch.jpg",
	} {
		t.Run("no false face/"+name, func(t *testing.T) {
			fixture, err := imageutil.Decode(testimages.Read(t, name))
			if err != nil {
				t.Fatal(err)
			}
			detections, err := model.Detect(context.Background(), fixture)
			if err != nil {
				t.Fatal(err)
			}
			if len(detections) != 0 {
				t.Fatalf("non-human fixture produced face detections: %+v", detections)
			}
		})
	}
}
