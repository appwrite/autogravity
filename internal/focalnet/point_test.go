package focalnet

import (
	"math"
	"testing"
)

func TestFromImportanceMassCentroid(t *testing.T) {
	// Left subject has more mass (6 vs 4). The mass centroid sits between
	// them; U²-Net's strongest-component rule would snap to the left blob.
	mass := make([]float32, 5*11)
	for _, point := range [][2]int{{1, 1}, {2, 1}, {1, 2}, {2, 2}, {1, 3}, {2, 3}} {
		mass[point[1]*11+point[0]] = 1
	}
	for _, point := range [][2]int{{8, 1}, {9, 1}, {8, 2}, {9, 2}} {
		mass[point[1]*11+point[0]] = 1
	}
	got, confidence, err := FromImportance(mass, 5, 11)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.X-0.43636363636363634) > 1e-9 || math.Abs(got.Y-0.46) > 1e-9 {
		t.Fatalf("FromImportance() = %+v, want mass centroid", got)
	}
	if confidence != 1 {
		t.Fatalf("confidence = %v, want 1", confidence)
	}
}

func TestFromImportanceEmptyAndUniform(t *testing.T) {
	empty, confidence, err := FromImportance(make([]float32, 8*4), 8, 4)
	if err != nil || empty.X != 0.5 || empty.Y != 0.5 || confidence != 0 {
		t.Fatalf("empty = %+v, %v, %v", empty, confidence, err)
	}

	uniform := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	got, confidence, err := FromImportance(uniform, 4, 4)
	if err != nil || math.Abs(got.X-0.5) > 1e-12 || math.Abs(got.Y-0.5) > 1e-12 || confidence != 1 {
		t.Fatalf("uniform = %+v, %v, %v", got, confidence, err)
	}
}

func TestFromImportanceLowerRightBlock(t *testing.T) {
	peak := make([]float32, 64*64)
	for y := 40; y < 48; y++ {
		for x := 48; x < 56; x++ {
			peak[y*64+x] = 1
		}
	}
	got, confidence, err := FromImportance(peak, 64, 64)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.X-0.8125) > 1e-12 || math.Abs(got.Y-0.6875) > 1e-12 || confidence != 1 {
		t.Fatalf("got %+v confidence %v", got, confidence)
	}
}

func TestFromImportanceIgnoresInvalidCells(t *testing.T) {
	values := []float32{1, float32(math.NaN()), -3, float32(math.Inf(1))}
	got, confidence, err := FromImportance(values, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got.X != 0.125 || got.Y != 0.5 || confidence != 1 {
		t.Fatalf("got %+v confidence %v", got, confidence)
	}
}

func TestFromImportanceRejectsInvalidDimensions(t *testing.T) {
	if _, _, err := FromImportance([]float32{1}, 2, 2); err == nil {
		t.Fatal("expected error")
	}
}
