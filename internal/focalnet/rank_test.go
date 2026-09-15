package focalnet

import (
	"testing"
)

func TestScoreCropFullImageRetainsAllMass(t *testing.T) {
	golden := loadRankingGolden(t)
	heatmap := []float32{0.2, 0.8, 0.4, 0.6}
	retained := ScoreCrop(heatmap, 2, 2, 0, 0, 4, 4, 4, 4)
	if retained != golden.ScoreCrop.Retained {
		t.Fatalf("ScoreCrop() = %v, want %v", retained, golden.ScoreCrop.Retained)
	}
}

func TestSelectRankedCropAppliesRetentionGateThenHumanScore(t *testing.T) {
	// Mass lives in the right half. The left crop keeps almost none of it,
	// so a higher human score cannot beat the retention gate.
	heatmap := make([]float32, MapSize*MapSize)
	for y := range MapSize {
		for x := MapSize / 2; x < MapSize; x++ {
			heatmap[y*MapSize+x] = 1
		}
	}
	candidates := [][4]float32{
		{0, 0, 0.4, 1},
		{0.5, 0, 1, 1},
		{0.45, 0, 1, 1},
	}
	scores := []float32{10, 1, 2}
	got, err := SelectRankedCrop(heatmap, MapSize, MapSize, candidates, scores, 100, 100)
	if err != nil {
		t.Fatal(err)
	}
	if got.Left != 45 || got.Top != 0 || got.Width != 55 || got.Height != 100 {
		t.Fatalf("crop = %+v, want the higher-scoring eligible right crop", got)
	}
	if got.Gravity.X < 0.7 || got.Gravity.X > 0.75 || got.Gravity.Y != 0.5 {
		t.Fatalf("gravity = %+v, want right-crop center", got.Gravity)
	}
	if got.HumanScore != 2 || got.Retention <= 0.9 {
		t.Fatalf("ranking = %+v", got)
	}
	if got.Confidence != 1 {
		t.Fatalf("confidence = %v, want peak 1", got.Confidence)
	}
}

func TestPixelCropRoundsToSourcePixels(t *testing.T) {
	left, top, width, height := PixelCrop([4]float32{0.25, 0.25, 0.75, 0.75}, 8, 8)
	if left != 2 || top != 2 || width != 4 || height != 4 {
		t.Fatalf("PixelCrop() = %d,%d %dx%d", left, top, width, height)
	}
}

func TestSelectRankedCropRejectsEmptyCandidates(t *testing.T) {
	if _, err := SelectRankedCrop(make([]float32, 4), 2, 2, nil, nil, 4, 4); err == nil {
		t.Fatal("expected error")
	}
}
