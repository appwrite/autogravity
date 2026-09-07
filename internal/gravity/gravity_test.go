package gravity

import (
	"image"
	"math"
	"testing"
)

func TestFromSaliency(t *testing.T) {
	tests := []struct {
		name       string
		mapData    []float32
		width      int
		height     int
		want       Point
		confidence float64
	}{
		{
			name:       "single point at top right",
			mapData:    []float32{0, 0, 1, 0, 0, 0, 0, 0, 0},
			width:      3,
			height:     3,
			want:       Point{X: 1, Y: 0},
			confidence: 1,
		},
		{
			name:       "strongest disconnected region",
			mapData:    []float32{1, 0, 3},
			width:      3,
			height:     1,
			want:       Point{X: 1, Y: 0.5},
			confidence: 1,
		},
		{
			name:       "weighted centroid within strongest region",
			mapData:    []float32{1, 1, 0, 0, 0, 0, 0.75, 0, 0},
			width:      3,
			height:     3,
			want:       Point{X: 0.25, Y: 0},
			confidence: 1,
		},
		{
			name:       "empty map falls back to center",
			mapData:    make([]float32, 8),
			width:      4,
			height:     2,
			want:       Point{X: 0.5, Y: 0.5},
			confidence: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, confidence, err := FromSaliency(tt.mapData, tt.width, tt.height)
			if err != nil {
				t.Fatalf("FromSaliency() error = %v", err)
			}
			if !closeEnough(got.X, tt.want.X) || !closeEnough(got.Y, tt.want.Y) {
				t.Fatalf("FromSaliency() = %+v, want %+v", got, tt.want)
			}
			if !closeEnough(confidence, tt.confidence) {
				t.Fatalf("confidence = %v, want %v", confidence, tt.confidence)
			}
		})
	}
}

func TestFromSaliencyRejectsInvalidDimensions(t *testing.T) {
	if _, _, err := FromSaliency([]float32{1}, 2, 2); err == nil {
		t.Fatal("FromSaliency() expected an error")
	}
}

func TestFromSaliencyRegionIgnoresPadding(t *testing.T) {
	mapData := make([]float32, 4*4)
	mapData[0] = 10      // Padding must not affect the result or confidence.
	mapData[1*4+3] = 0.5 // Top-right of the image content.

	got, confidence, err := FromSaliencyRegion(mapData, 4, 4, image.Rect(0, 1, 4, 3))
	if err != nil {
		t.Fatal(err)
	}
	if want := (Point{X: 1, Y: 0}); got != want {
		t.Fatalf("FromSaliencyRegion() = %+v, want %+v", got, want)
	}
	if confidence != 0.5 {
		t.Fatalf("confidence = %v, want 0.5", confidence)
	}
}

func TestFromSaliencyRegionRejectsInvalidRegion(t *testing.T) {
	if _, _, err := FromSaliencyRegion(make([]float32, 4), 2, 2, image.Rect(-1, 0, 1, 1)); err == nil {
		t.Fatal("FromSaliencyRegion() expected an error")
	}
}

func closeEnough(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}
