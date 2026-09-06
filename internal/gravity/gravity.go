// Package gravity calculates focal points from saliency maps.
package gravity

import (
	"errors"
	"image"
	"math"
)

// Point is a normalized coordinate in an image.
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// FromSaliency returns the saliency-weighted centroid and the peak saliency
// value as confidence. Values that are negative, NaN, or infinite contribute
// no weight. A map with no usable saliency falls back to the image center.
func FromSaliency(saliency []float32, width, height int) (Point, float64, error) {
	return FromSaliencyRegion(saliency, width, height, image.Rect(0, 0, width, height))
}

// FromSaliencyRegion calculates a focal point from a rectangular image region
// within a larger saliency map, ignoring any surrounding letterbox padding.
func FromSaliencyRegion(saliency []float32, width, height int, region image.Rectangle) (Point, float64, error) {
	if width <= 0 || height <= 0 || len(saliency) != width*height {
		return Point{}, 0, errors.New("invalid saliency map dimensions")
	}
	mapBounds := image.Rect(0, 0, width, height)
	if region.Empty() || region.Intersect(mapBounds) != region {
		return Point{}, 0, errors.New("invalid saliency map region")
	}

	var total, weightedX, weightedY, peak float64
	for y := region.Min.Y; y < region.Max.Y; y++ {
		for x := region.Min.X; x < region.Max.X; x++ {
			weight := float64(saliency[y*width+x])
			if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
				continue
			}
			total += weight
			weightedX += float64(x-region.Min.X) * weight
			weightedY += float64(y-region.Min.Y) * weight
			if weight > peak {
				peak = weight
			}
		}
	}

	confidence := clamp01(peak)
	if total == 0 {
		return Point{X: 0.5, Y: 0.5}, confidence, nil
	}

	x, y := 0.5, 0.5
	if region.Dx() > 1 {
		x = weightedX / total / float64(region.Dx()-1)
	}
	if region.Dy() > 1 {
		y = weightedY / total / float64(region.Dy()-1)
	}

	return Point{X: clamp01(x), Y: clamp01(y)}, confidence, nil
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
