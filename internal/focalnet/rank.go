package focalnet

import (
	"errors"
	"math"

	"autogravity/internal/gravity"
)

// PixelCrop converts a normalized [left, top, right, bottom] box to integer
// pixel coordinates in the oriented source image.
func PixelCrop(box [4]float32, width, height int) (left, top, cropWidth, cropHeight int) {
	left = max(0, min(width-1, int(float64(box[0])*float64(width)+0.5)))
	top = max(0, min(height-1, int(float64(box[1])*float64(height)+0.5)))
	right := max(left+1, min(width, int(float64(box[2])*float64(width)+0.5)))
	bottom := max(top+1, min(height, int(float64(box[3])*float64(height)+0.5)))
	return left, top, right - left, bottom - top
}

// ScoreCrop is the fraction of heatmap mass inside a pixel crop.
func ScoreCrop(heatmap []float32, rows, cols, left, top, width, height, imageWidth, imageHeight int) float64 {
	var total float64
	clean := make([]float64, len(heatmap))
	for i, value := range heatmap {
		weight := float64(value)
		if weight > 0 && !math.IsNaN(weight) && !math.IsInf(weight, 0) {
			clean[i] = weight
			total += weight
		}
	}
	if total == 0 || imageWidth <= 0 || imageHeight <= 0 {
		return 0
	}

	xs := linspace(0, float64(imageWidth), cols+1)
	ys := linspace(0, float64(imageHeight), rows+1)
	wx := make([]float64, cols)
	wy := make([]float64, rows)
	cropRight := float64(left + width)
	cropBottom := float64(top + height)
	cellW := float64(imageWidth) / float64(cols)
	cellH := float64(imageHeight) / float64(rows)
	for i := range cols {
		wx[i] = max(0, min(xs[i+1], cropRight)-max(xs[i], float64(left))) / cellW
	}
	for i := range rows {
		wy[i] = max(0, min(ys[i+1], cropBottom)-max(ys[i], float64(top))) / cellH
	}

	var retained float64
	for y := range rows {
		if wy[y] == 0 {
			continue
		}
		var row float64
		for x := range cols {
			row += clean[y*cols+x] * wx[x]
		}
		retained += row * wy[y]
	}
	return clamp01(retained / total)
}

// RankedCrop is the human-selected crop and the gravity derived from it.
type RankedCrop struct {
	Left, Top, Width, Height int
	Gravity                  gravity.Point
	Confidence               float64
	HumanScore               float64
	Retention                float64
}

// SelectRankedCrop keeps candidates within RetentionTolerance of the best
// importance retention, then picks the highest human score among them.
func SelectRankedCrop(
	heatmap []float32,
	rows, cols int,
	candidates [][4]float32,
	scores []float32,
	imageWidth, imageHeight int,
) (RankedCrop, error) {
	if len(candidates) == 0 {
		return RankedCrop{}, errors.New("no crop candidates")
	}
	if len(scores) < len(candidates) {
		return RankedCrop{}, errors.New("not enough crop scores")
	}

	retention := make([]float64, len(candidates))
	var maximum float64
	for i, box := range candidates {
		left, top, width, height := PixelCrop(box, imageWidth, imageHeight)
		retention[i] = ScoreCrop(heatmap, rows, cols, left, top, width, height, imageWidth, imageHeight)
		if retention[i] > maximum {
			maximum = retention[i]
		}
	}

	selected := -1
	bestScore := math.Inf(-1)
	for i, retained := range retention {
		if retained < maximum-RetentionTolerance {
			continue
		}
		if float64(scores[i]) > bestScore {
			bestScore = float64(scores[i])
			selected = i
		}
	}
	if selected < 0 {
		return RankedCrop{}, errors.New("no eligible crop candidate")
	}

	left, top, width, height := PixelCrop(candidates[selected], imageWidth, imageHeight)
	_, confidence, err := FromImportance(heatmap, rows, cols)
	if err != nil {
		return RankedCrop{}, err
	}
	return RankedCrop{
		Left:       left,
		Top:        top,
		Width:      width,
		Height:     height,
		Gravity:    cropCenter(left, top, width, height, imageWidth, imageHeight),
		Confidence: confidence,
		HumanScore: float64(scores[selected]),
		Retention:  retention[selected],
	}, nil
}

func cropCenter(left, top, width, height, imageWidth, imageHeight int) gravity.Point {
	return gravity.Point{
		X: clamp01((float64(left) + float64(width)/2) / float64(imageWidth)),
		Y: clamp01((float64(top) + float64(height)/2) / float64(imageHeight)),
	}
}
