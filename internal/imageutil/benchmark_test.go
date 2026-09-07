package imageutil

import (
	"image"
	"testing"

	"autogravity/internal/testimages"
)

var preparedTensor []float32

func BenchmarkPrepareInto(b *testing.B) {
	for _, name := range []string{"portrait.jpg", "rose-alpha.webp"} {
		b.Run(name, func(b *testing.B) {
			img, err := Decode(testimages.Read(b, name))
			if err != nil {
				b.Fatal(err)
			}
			dst := make([]float32, 3*320*320)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := PrepareInto(dst, img, 320, 320); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkPrepare(b *testing.B) {
	for _, name := range []string{"portrait.jpg", "rose-alpha.webp"} {
		b.Run(name, func(b *testing.B) {
			img, err := Decode(testimages.Read(b, name))
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				preparedTensor, _, err = Prepare(img, 320, 320)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// This isolates normalization from resizing and includes every alpha value.
func BenchmarkPrepareAlreadySized(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 320, 320))
	for i := range img.Pix {
		img.Pix[i] = byte(i / 7)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var err error
		preparedTensor, _, err = Prepare(img, 320, 320)
		if err != nil {
			b.Fatal(err)
		}
	}
}
