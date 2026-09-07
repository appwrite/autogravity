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

// FromSaliency returns the centroid of the strongest connected salient region
// and the peak saliency value as confidence. Values that are negative, NaN, or
// infinite contribute no weight. A map with no usable saliency falls back to
// the image center.
func FromSaliency(saliency []float32, width, height int) (Point, float64, error) {
	return FromSaliencyRegion(saliency, width, height, image.Rect(0, 0, width, height))
}

// FromSaliencyRegion calculates a focal point from a rectangular image region
// within a larger saliency map, ignoring any surrounding letterbox padding.
// Pixels at least half as salient as the peak are grouped into 8-connected
// components. The component with the greatest total saliency is selected so
// separated subjects do not produce a focal point in the empty space between
// them.
func FromSaliencyRegion(saliency []float32, width, height int, region image.Rectangle) (Point, float64, error) {
	if width <= 0 || height <= 0 || len(saliency) != width*height {
		return Point{}, 0, errors.New("invalid saliency map dimensions")
	}
	mapBounds := image.Rect(0, 0, width, height)
	if region.Empty() || region.Intersect(mapBounds) != region {
		return Point{}, 0, errors.New("invalid saliency map region")
	}

	var peak float64
	for y := region.Min.Y; y < region.Max.Y; y++ {
		for x := region.Min.X; x < region.Max.X; x++ {
			weight := float64(saliency[y*width+x])
			if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
				continue
			}
			if weight > peak {
				peak = weight
			}
		}
	}

	confidence := clamp01(peak)
	if peak == 0 {
		return Point{X: 0.5, Y: 0.5}, confidence, nil
	}

	type component struct {
		total                float64
		weightedX, weightedY float64
	}
	threshold := peak * 0.5
	visited := make([]bool, len(saliency))
	var strongest component
	directions := [...]image.Point{
		{X: -1, Y: -1}, {X: 0, Y: -1}, {X: 1, Y: -1},
		{X: -1, Y: 0}, {X: 1, Y: 0},
		{X: -1, Y: 1}, {X: 0, Y: 1}, {X: 1, Y: 1},
	}
	for y := region.Min.Y; y < region.Max.Y; y++ {
		for x := region.Min.X; x < region.Max.X; x++ {
			index := y*width + x
			if visited[index] || !usableAtThreshold(saliency[index], threshold) {
				continue
			}
			visited[index] = true
			queue := []image.Point{{X: x, Y: y}}
			var current component
			for len(queue) > 0 {
				point := queue[len(queue)-1]
				queue = queue[:len(queue)-1]
				weight := float64(saliency[point.Y*width+point.X])
				current.total += weight
				current.weightedX += float64(point.X-region.Min.X) * weight
				current.weightedY += float64(point.Y-region.Min.Y) * weight

				for _, direction := range directions {
					neighbor := point.Add(direction)
					if !neighbor.In(region) {
						continue
					}
					neighborIndex := neighbor.Y*width + neighbor.X
					if visited[neighborIndex] || !usableAtThreshold(saliency[neighborIndex], threshold) {
						continue
					}
					visited[neighborIndex] = true
					queue = append(queue, neighbor)
				}
			}
			if current.total > strongest.total {
				strongest = current
			}
		}
	}

	x, y := 0.5, 0.5
	if region.Dx() > 1 {
		x = strongest.weightedX / strongest.total / float64(region.Dx()-1)
	}
	if region.Dy() > 1 {
		y = strongest.weightedY / strongest.total / float64(region.Dy()-1)
	}

	return Point{X: clamp01(x), Y: clamp01(y)}, confidence, nil
}

func usableAtThreshold(value float32, threshold float64) bool {
	weight := float64(value)
	return weight >= threshold && !math.IsNaN(weight) && !math.IsInf(weight, 0)
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
