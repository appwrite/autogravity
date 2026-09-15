package focalnet

import (
	"math"
	"testing"
)

func TestPaddingCannotAffectRestoredHeatmap(t *testing.T) {
	for _, dimensions := range [][2]int{{199, 301}, {640, 320}, {1, 10000}, {901, 109}, {256, 256}} {
		box, err := Fit(dimensions[0], dimensions[1], InputSize)
		if err != nil {
			t.Fatal(err)
		}
		coverage := box.Coverage(MapSize)
		heatmap := make([]float32, MapSize*MapSize)
		for i, weight := range coverage {
			if weight > 0 {
				heatmap[i] = 0.4
			} else {
				heatmap[i] = 1e6
			}
		}
		restored, err := RestoreMap(heatmap, MapSize, box, MapSize)
		if err != nil {
			t.Fatal(err)
		}
		for i, value := range restored {
			if math.Abs(float64(value)-0.4) > 1e-5 {
				t.Fatalf("%dx%d restored[%d] = %v, want 0.4", dimensions[0], dimensions[1], i, value)
			}
		}
		var coverageSum float64
		for _, weight := range coverage {
			coverageSum += float64(weight)
		}
		wantCells := float64(box.Width*box.Height) / 16
		if math.Abs(coverageSum-wantCells) > 1e-4 {
			t.Fatalf("%dx%d coverage sum = %v, want %v", dimensions[0], dimensions[1], coverageSum, wantCells)
		}
	}
}

func TestRestoreMapRejectsInvalidInput(t *testing.T) {
	box := Letterbox{Size: 256, Width: 128, Height: 256}
	if _, err := RestoreMap(make([]float32, 8), 4, box, 4); err == nil {
		t.Fatal("expected error for wrong length")
	}
}
