// Package testimages provides checked-in image fixtures shared by tests.
package testimages

import (
	"embed"
	"image"
	"testing"
)

//go:embed testdata/*.png testdata/*.jpg testdata/*.webp
var files embed.FS

type Fixture struct {
	Name        string
	ContentType string
	Size        image.Point
	Content     image.Rectangle // Expected letterbox region in a 320x320 tensor.
}

var All = []Fixture{
	{"rose.png", "image/png", image.Pt(400, 301), image.Rect(0, 39, 320, 280)},
	{"rose-lossless.webp", "image/webp", image.Pt(400, 301), image.Rect(0, 39, 320, 280)},
	{"rose-lossy.webp", "image/webp", image.Pt(400, 301), image.Rect(0, 39, 320, 280)},
	{"rose-alpha.webp", "image/webp", image.Pt(400, 301), image.Rect(0, 39, 320, 280)},
	{"portrait.jpg", "image/jpeg", image.Pt(280, 360), image.Rect(35, 0, 284, 320)},
}

func Read(t testing.TB, name string) []byte {
	t.Helper()
	data, err := files.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
