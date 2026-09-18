package focalnet

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type rankingGolden struct {
	Candidates []struct {
		Width       int          `json:"width"`
		Height      int          `json:"height"`
		AspectRatio float64      `json:"aspect_ratio"`
		Boxes       [][4]float32 `json:"boxes"`
	} `json:"candidates"`
	Content struct {
		Size       int       `json:"size"`
		Left       int       `json:"left"`
		Top        int       `json:"top"`
		Width      int       `json:"width"`
		Height     int       `json:"height"`
		Normalized []float32 `json:"normalized"`
	} `json:"content"`
	ScoreCrop struct {
		Retained float64 `json:"retained"`
	} `json:"score_crop"`
}

func loadRankingGolden(t *testing.T) rankingGolden {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "ranking.json"))
	if err != nil {
		t.Fatal(err)
	}
	var golden rankingGolden
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	return golden
}

func TestGenerateCandidatesMatchesPython(t *testing.T) {
	golden := loadRankingGolden(t)
	for _, tc := range golden.Candidates {
		got, err := GenerateCandidates(tc.Width, tc.Height, tc.AspectRatio)
		if err != nil {
			t.Fatalf("GenerateCandidates(%d,%d,%g): %v", tc.Width, tc.Height, tc.AspectRatio, err)
		}
		if len(got) != len(tc.Boxes) {
			t.Fatalf("GenerateCandidates(%d,%d,%g) len=%d, want %d", tc.Width, tc.Height, tc.AspectRatio, len(got), len(tc.Boxes))
		}
		for i, box := range got {
			if box != tc.Boxes[i] {
				t.Fatalf("GenerateCandidates(%d,%d,%g)[%d] = %v, want %v", tc.Width, tc.Height, tc.AspectRatio, i, box, tc.Boxes[i])
			}
		}
	}
}

func TestLetterboxContentMatchesPython(t *testing.T) {
	golden := loadRankingGolden(t)
	box := Letterbox{
		Size:   golden.Content.Size,
		Left:   golden.Content.Left,
		Top:    golden.Content.Top,
		Width:  golden.Content.Width,
		Height: golden.Content.Height,
	}
	got := box.Content()
	if len(got) != len(golden.Content.Normalized) {
		t.Fatalf("Content() len=%d, want %d", len(got), len(golden.Content.Normalized))
	}
	for i, value := range got {
		if value != golden.Content.Normalized[i] {
			t.Fatalf("Content()[%d] = %v, want %v", i, value, golden.Content.Normalized[i])
		}
	}
}

func TestPadBoxesZeroFillsUnusedSlots(t *testing.T) {
	padded := PadBoxes([][4]float32{{0.1, 0.2, 0.3, 0.4}})
	if len(padded) != MaxCandidates*4 {
		t.Fatalf("len = %d, want %d", len(padded), MaxCandidates*4)
	}
	if padded[0] != 0.1 || padded[1] != 0.2 || padded[2] != 0.3 || padded[3] != 0.4 {
		t.Fatalf("first box = %v", padded[:4])
	}
	for _, value := range padded[4:] {
		if value != 0 {
			t.Fatalf("padding slot = %v, want 0", value)
		}
	}
}

func TestParseAspectRatio(t *testing.T) {
	for _, tc := range []struct {
		in      string
		want    float64
		invalid bool
	}{
		{"", 1, false},
		{"1", 1, false},
		{"1.5", 1.5, false},
		{"16:9", 16.0 / 9.0, false},
		{"4:5", 0.8, false},
		{" 16:9 ", 16.0 / 9.0, false},
		{"0", 0, true},
		{"-1", 0, true},
		{"16:0", 0, true},
		{"nope", 0, true},
		{"1:2:3", 0, true},
	} {
		got, err := ParseAspectRatio(tc.in)
		if tc.invalid {
			if err == nil {
				t.Fatalf("ParseAspectRatio(%q) = %v, want error", tc.in, got)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Fatalf("ParseAspectRatio(%q) = %v, %v; want %v", tc.in, got, err, tc.want)
		}
	}
}

func TestGenerateCandidatesRejectsInvalidInputs(t *testing.T) {
	if _, err := GenerateCandidates(0, 10, 1); err == nil {
		t.Fatal("expected error for non-positive size")
	}
	if _, err := GenerateCandidates(10, 10, 0); err == nil {
		t.Fatal("expected error for non-positive aspect ratio")
	}
}
