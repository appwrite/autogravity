package focalnet

import (
	"errors"
	"math"
)

// RestoreMap removes letterbox padding from a square importance map and
// returns a map in normalized oriented-image coordinates. size is the
// restored grid edge; the Python runtime uses MapSize.
func RestoreMap(values []float32, grid int, box Letterbox, size int) ([]float32, error) {
	if grid <= 0 || size <= 0 || len(values) != grid*grid {
		return nil, errors.New("expected a square 2D map")
	}
	if box.Size <= 0 || box.Width <= 0 || box.Height <= 0 {
		return nil, errors.New("invalid letterbox")
	}

	xs := make([]float64, size)
	ys := make([]float64, size)
	for i := range size {
		center := (float64(i) + 0.5) / float64(size)
		xs[i] = (float64(box.Left)+center*float64(box.Width))/float64(box.Size)*float64(grid) - 0.5
		ys[i] = (float64(box.Top)+center*float64(box.Height))/float64(box.Size)*float64(grid) - 0.5
	}

	coverage := box.Coverage(grid)
	valid := make([]float32, len(coverage))
	masked := make([]float32, len(values))
	for i, weight := range coverage {
		if weight > 0 {
			valid[i] = 1
			masked[i] = values[i]
		}
	}

	numerator := sample(masked, grid, grid, xs, ys)
	denominator := sample(valid, grid, grid, xs, ys)
	out := make([]float32, size*size)
	for i := range out {
		out[i] = numerator[i] / float32(max(float64(denominator[i]), 1e-8))
	}
	return out, nil
}

func sample(values []float32, rows, cols int, xs, ys []float64) []float32 {
	out := make([]float32, len(ys)*len(xs))
	maxX := float64(cols - 1)
	maxY := float64(rows - 1)
	for i, y := range ys {
		y = clamp(y, 0, maxY)
		y0 := int(math.Floor(y))
		y1 := min(y0+1, rows-1)
		wy := y - float64(y0)
		for j, x := range xs {
			x = clamp(x, 0, maxX)
			x0 := int(math.Floor(x))
			x1 := min(x0+1, cols-1)
			wx := x - float64(x0)
			top := float64(values[y0*cols+x0])*(1-wx) + float64(values[y0*cols+x1])*wx
			bottom := float64(values[y1*cols+x0])*(1-wx) + float64(values[y1*cols+x1])*wx
			out[i*len(xs)+j] = float32(top*(1-wy) + bottom*wy)
		}
	}
	return out
}

func clamp(value, lo, hi float64) float64 {
	if value < lo {
		return lo
	}
	if value > hi {
		return hi
	}
	return value
}
