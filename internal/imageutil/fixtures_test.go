package imageutil

import (
	"bytes"
	"image"
	"math"
	"testing"

	"autogravity/internal/testimages"
	"github.com/disintegration/imaging"
)

func TestDecodeAndPrepareFixtures(t *testing.T) {
	for _, fixture := range testimages.All {
		t.Run(fixture.Name, func(t *testing.T) {
			data := testimages.Read(t, fixture.Name)
			original := bytes.Clone(data)
			img, err := Decode(data)
			if err != nil {
				t.Fatal(err)
			}
			if img.Bounds().Size() != fixture.Size {
				t.Fatalf("size = %v, want %v", img.Bounds().Size(), fixture.Size)
			}
			tensor, content, err := Prepare(img, 320, 320)
			if err != nil {
				t.Fatal(err)
			}
			if content != fixture.Content {
				t.Fatalf("content = %v, want %v", content, fixture.Content)
			}
			if len(tensor) != 3*320*320 {
				t.Fatalf("tensor length = %d", len(tensor))
			}
			for c := 0; c < 3; c++ {
				for y := 0; y < 320; y++ {
					for x := 0; x < 320; x++ {
						v := tensor[c*320*320+y*320+x]
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < -2.2 || v > 2.7 {
							t.Fatalf("invalid normalized value %v", v)
						}
						if !image.Pt(x, y).In(content) && v != 0 {
							t.Fatalf("padding at %d,%d channel %d = %v", x, y, c, v)
						}
					}
				}
			}
			if !bytes.Equal(data, original) {
				t.Fatal("input bytes were modified")
			}

			// Without resizing, every channel must match the decoded photograph,
			// including premultiplied color values at transparent pixels.
			tensor, _, err = Prepare(img, fixture.Size.X, fixture.Size.Y)
			if err != nil {
				t.Fatal(err)
			}
			pixels := imaging.Clone(img) // Match the decoder's 8-bit color conversion.
			mean := [3]float64{0.485, 0.456, 0.406}
			std := [3]float64{0.229, 0.224, 0.225}
			for y := 0; y < fixture.Size.Y; y++ {
				for x := 0; x < fixture.Size.X; x++ {
					r, g, b, _ := pixels.At(x, y).RGBA()
					for c, value := range []uint32{r, g, b} {
						want := (float64(value)/65535 - mean[c]) / std[c]
						got := tensor[c*fixture.Size.X*fixture.Size.Y+y*fixture.Size.X+x]
						if math.Abs(float64(got)-want) > 0.00001 {
							t.Fatalf("pixel %d,%d channel %d = %v, want %v", x, y, c, got, want)
						}
					}
				}
			}
		})
	}
}

func TestDecodeTruncatedFixtures(t *testing.T) {
	for _, fixture := range testimages.All {
		t.Run(fixture.Name, func(t *testing.T) {
			data := testimages.Read(t, fixture.Name)
			for _, length := range []int{16, len(data) / 2} {
				if _, err := Decode(data[:length]); err == nil {
					t.Fatalf("accepted %s truncated to %d bytes", fixture.Name, length)
				}
			}
		})
	}
}

func TestDecodeAllEXIFOrientations(t *testing.T) {
	data := testimages.Read(t, "portrait.jpg")
	source, err := Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	source = imaging.Clone(source)
	w, h := source.Bounds().Dx(), source.Bounds().Dy()
	tests := []struct {
		name        string
		orientation uint16
		size        image.Point
		sourcePoint func(int, int) (int, int)
	}{
		{"normal", 1, image.Pt(w, h), func(x, y int) (int, int) { return x, y }},
		{"mirror horizontal", 2, image.Pt(w, h), func(x, y int) (int, int) { return w - 1 - x, y }},
		{"rotate 180", 3, image.Pt(w, h), func(x, y int) (int, int) { return w - 1 - x, h - 1 - y }},
		{"mirror vertical", 4, image.Pt(w, h), func(x, y int) (int, int) { return x, h - 1 - y }},
		{"transpose", 5, image.Pt(h, w), func(x, y int) (int, int) { return y, x }},
		{"rotate 90", 6, image.Pt(h, w), func(x, y int) (int, int) { return y, h - 1 - x }},
		{"transverse", 7, image.Pt(h, w), func(x, y int) (int, int) { return w - 1 - y, h - 1 - x }},
		{"rotate 270", 8, image.Pt(h, w), func(x, y int) (int, int) { return w - 1 - y, x }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Decode(withEXIFOrientation(data, tt.orientation))
			if err != nil {
				t.Fatal(err)
			}
			got = imaging.Clone(got)
			if got.Bounds().Size() != tt.size {
				t.Fatalf("size = %v, want %v", got.Bounds().Size(), tt.size)
			}
			for y := 0; y < tt.size.Y; y++ {
				for x := 0; x < tt.size.X; x++ {
					sx, sy := tt.sourcePoint(x, y)
					r, g, b, a := got.At(x, y).RGBA()
					wr, wg, wb, wa := source.At(sx, sy).RGBA()
					if r != wr || g != wg || b != wb || a != wa {
						t.Fatalf("oriented pixel %d,%d does not match source %d,%d", x, y, sx, sy)
					}
				}
			}
		})
	}
}
