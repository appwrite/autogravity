// Package focalnet runs the distilled FocalNet importance model and restores
// its 64×64 map into oriented-image coordinates.
package focalnet

import (
	"errors"
	"image"
	"math"
)

const (
	// InputSize is the letterboxed RGB tensor edge length.
	InputSize = 256
	// MapSize is the square importance-map edge length.
	MapSize = 64
)

// Letterbox is the content rectangle inside a square model input.
type Letterbox struct {
	Size   int
	Left   int
	Top    int
	Width  int
	Height int
}

// Fit aspect-fits an image into a square canvas. Rounding matches
// imageutil.Prepare, including ties away from zero for positive values.
func Fit(width, height, size int) (Letterbox, error) {
	if min(width, height, size) <= 0 {
		return Letterbox{}, errors.New("image and input dimensions must be positive")
	}
	scale := math.Min(float64(size)/float64(width), float64(size)/float64(height))
	fitWidth := max(1, min(size, int(math.Round(float64(width)*scale))))
	fitHeight := max(1, min(size, int(math.Round(float64(height)*scale))))
	return Letterbox{
		Size:   size,
		Left:   (size - fitWidth) / 2,
		Top:    (size - fitHeight) / 2,
		Width:  fitWidth,
		Height: fitHeight,
	}, nil
}

// FromContent reconstructs the letterbox used to produce a prepared tensor.
func FromContent(content image.Rectangle, size int) Letterbox {
	return Letterbox{
		Size:   size,
		Left:   content.Min.X,
		Top:    content.Min.Y,
		Width:  content.Dx(),
		Height: content.Dy(),
	}
}

// Coverage is the fraction of each map cell occupied by image content.
func (b Letterbox) Coverage(mapSize int) []float32 {
	if mapSize <= 0 || b.Size <= 0 {
		return nil
	}
	edges := make([]float64, mapSize+1)
	for i := range edges {
		edges[i] = float64(b.Size) * float64(i) / float64(mapSize)
	}
	x := make([]float64, mapSize)
	y := make([]float64, mapSize)
	right := float64(b.Left + b.Width)
	bottom := float64(b.Top + b.Height)
	for i := range mapSize {
		x[i] = max(0, min(edges[i+1], right)-max(edges[i], float64(b.Left)))
		y[i] = max(0, min(edges[i+1], bottom)-max(edges[i], float64(b.Top)))
	}
	cell := float64(b.Size) / float64(mapSize)
	area := cell * cell
	out := make([]float32, mapSize*mapSize)
	for row := range mapSize {
		for col := range mapSize {
			out[row*mapSize+col] = float32(y[row] * x[col] / area)
		}
	}
	return out
}
