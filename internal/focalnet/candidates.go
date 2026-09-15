package focalnet

import (
	"errors"
	"math"
	"slices"
	"strconv"
	"strings"
)

var candidateScales = []float64{1.0, 0.9, 0.8, 0.7, 0.65}

const candidatePositions = 5

// GenerateCandidates returns exact-ratio source-normalized crops
// [left, top, right, bottom] at several positions and zoom levels.
func GenerateCandidates(imageWidth, imageHeight int, aspectRatio float64) ([][4]float32, error) {
	if min(imageWidth, imageHeight) <= 0 {
		return nil, errors.New("image dimensions must be positive")
	}
	if !isPositiveFinite(aspectRatio) {
		return nil, errors.New("aspect ratio must be finite and positive")
	}

	sourceRatio := float64(imageWidth) / float64(imageHeight)
	var maximumWidth, maximumHeight float64
	if sourceRatio > aspectRatio {
		maximumWidth, maximumHeight = aspectRatio/sourceRatio, 1
	} else {
		maximumWidth, maximumHeight = 1, sourceRatio/aspectRatio
	}

	raw := make([][4]float32, 0, 125)
	for _, scale := range candidateScales {
		width, height := maximumWidth*scale, maximumHeight*scale
		xCount, yCount := 1, 1
		if width < 1-1e-7 {
			xCount = candidatePositions
		}
		if height < 1-1e-7 {
			yCount = candidatePositions
		}
		for _, top := range linspace(0, 1-height, yCount) {
			for _, left := range linspace(0, 1-width, xCount) {
				raw = append(raw, [4]float32{
					round7(left),
					round7(top),
					round7(left + width),
					round7(top + height),
				})
			}
		}
	}
	slices.SortFunc(raw, compareBox)
	unique := raw[:0]
	for _, box := range raw {
		if len(unique) == 0 || unique[len(unique)-1] != box {
			unique = append(unique, box)
		}
	}
	unique = slices.Clip(unique)
	if len(unique) > MaxCandidates {
		return nil, errors.New("generated more crop candidates than the model accepts")
	}
	return unique, nil
}

// ParseAspectRatio accepts a positive float or W:H pair. An empty value is 1:1.
func ParseAspectRatio(value string) (float64, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 1, nil
	}
	if strings.Contains(value, ":") {
		parts := strings.Split(value, ":")
		if len(parts) != 2 {
			return 0, errInvalidAspectRatio
		}
		width, widthErr := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
		height, heightErr := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if widthErr != nil || heightErr != nil || !isPositiveFinite(width) || !isPositiveFinite(height) {
			return 0, errInvalidAspectRatio
		}
		return width / height, nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || !isPositiveFinite(parsed) {
		return 0, errInvalidAspectRatio
	}
	return parsed, nil
}

var errInvalidAspectRatio = errors.New("aspect ratio must be a positive number or W:H, for example 16:9 or 1.5")

// PadBoxes copies candidates into the model's [1,128,4] input, zero-filling
// unused slots.
func PadBoxes(candidates [][4]float32) []float32 {
	out := make([]float32, MaxCandidates*4)
	for i, box := range candidates {
		copy(out[i*4:], box[:])
	}
	return out
}

func linspace(start, stop float64, count int) []float64 {
	if count <= 1 {
		return []float64{start}
	}
	out := make([]float64, count)
	step := (stop - start) / float64(count-1)
	for i := range count {
		out[i] = start + step*float64(i)
	}
	return out
}

func round7(value float64) float32 {
	// Scale in float32 and round half-to-even so unique-ing matches np.round(float32, 7).
	as32 := float32(value)
	return float32(math.RoundToEven(float64(as32*1e7))) / 1e7
}

func compareBox(a, b [4]float32) int {
	for i := range 4 {
		switch {
		case a[i] < b[i]:
			return -1
		case a[i] > b[i]:
			return 1
		}
	}
	return 0
}

func isPositiveFinite(value float64) bool {
	return value > 0 && !math.IsInf(value, 0) && !math.IsNaN(value)
}
