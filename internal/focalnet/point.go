package focalnet

import (
	"errors"
	"math"

	"autogravity/internal/gravity"
)

// FromImportance returns the mass centroid of an importance map. Invalid or
// non-positive cells contribute no weight. An empty map falls back to center.
// Confidence is the peak activation, clamped to [0, 1].
func FromImportance(heatmap []float32, rows, cols int) (gravity.Point, float64, error) {
	if rows <= 0 || cols <= 0 || len(heatmap) != rows*cols {
		return gravity.Point{}, 0, errors.New("invalid importance map dimensions")
	}

	var total, weightedX, weightedY, peak float64
	for y := range rows {
		for x := range cols {
			weight := float64(heatmap[y*cols+x])
			if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
				continue
			}
			if weight > peak {
				peak = weight
			}
			total += weight
			weightedX += weight * (float64(x) + 0.5) / float64(cols)
			weightedY += weight * (float64(y) + 0.5) / float64(rows)
		}
	}

	confidence := clamp01(peak)
	if total == 0 {
		return gravity.Point{X: 0.5, Y: 0.5}, confidence, nil
	}
	return gravity.Point{X: weightedX / total, Y: weightedY / total}, confidence, nil
}

func clamp01(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}
