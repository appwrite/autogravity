package focalnet

import (
	"encoding/json"
	"image"
	"os"
	"path/filepath"
	"testing"

	"autogravity/internal/imageutil"
)

type letterboxGolden struct {
	Width     int `json:"width"`
	Height    int `json:"height"`
	Size      int `json:"size"`
	Left      int `json:"left"`
	Top       int `json:"top"`
	FitWidth  int `json:"fit_width"`
	FitHeight int `json:"fit_height"`
}

func TestFitMatchesPythonLetterboxes(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "letterboxes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []letterboxGolden
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		got, err := Fit(tc.Width, tc.Height, tc.Size)
		if err != nil {
			t.Fatalf("Fit(%d,%d,%d): %v", tc.Width, tc.Height, tc.Size, err)
		}
		want := Letterbox{Size: tc.Size, Left: tc.Left, Top: tc.Top, Width: tc.FitWidth, Height: tc.FitHeight}
		if got != want {
			t.Fatalf("Fit(%d,%d,%d) = %+v, want %+v", tc.Width, tc.Height, tc.Size, got, want)
		}
	}
}

func TestPrepareLetterboxMatchesFit(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "letterboxes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []letterboxGolden
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		img := image.NewNRGBA(image.Rect(0, 0, tc.Width, tc.Height))
		_, content, err := imageutil.Prepare(img, tc.Size, tc.Size)
		if err != nil {
			t.Fatalf("Prepare(%d,%d): %v", tc.Width, tc.Height, err)
		}
		box, err := Fit(tc.Width, tc.Height, tc.Size)
		if err != nil {
			t.Fatal(err)
		}
		want := image.Rect(box.Left, box.Top, box.Left+box.Width, box.Top+box.Height)
		if content != want {
			t.Fatalf("Prepare content %v, Fit %v", content, want)
		}
	}
}

func TestFitRejectsNonPositiveDimensions(t *testing.T) {
	if _, err := Fit(0, 10, 256); err == nil {
		t.Fatal("expected error")
	}
}

func TestRoundingMatchesGoHalfUp(t *testing.T) {
	box, err := Fit(512, 257, 256)
	if err != nil {
		t.Fatal(err)
	}
	if box.Height != 129 {
		t.Fatalf("height = %d, want 129", box.Height)
	}
}
