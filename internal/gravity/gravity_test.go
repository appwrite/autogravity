package gravity

import (
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
			name:       "weighted centroid",
			mapData:    []float32{1, 0, 0, 3},
			width:      2,
			height:     2,
			want:       Point{X: 0.75, Y: 0.75},
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

func closeEnough(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}
