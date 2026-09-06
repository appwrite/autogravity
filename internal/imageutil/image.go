// Package imageutil decodes and prepares images for the saliency model.
package imageutil

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"

	"github.com/disintegration/imaging"
	_ "golang.org/x/image/webp"
)

const MaxPixels = 20_000_000

var (
	ErrUnsupportedFormat = errors.New("unsupported image format")
	ErrImageTooLarge     = errors.New("image dimensions are too large")
)

// Decode decodes JPEG, PNG, or WebP data and applies any EXIF orientation.
func Decode(data []byte) (image.Image, error) {
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		if errors.Is(err, image.ErrFormat) {
			return nil, fmt.Errorf("%w: unknown image format", ErrUnsupportedFormat)
		}
		return nil, fmt.Errorf("decode image header: %w", err)
	}
	if format != "jpeg" && format != "png" && format != "webp" {
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedFormat, format)
	}
	if config.Width <= 0 || config.Height <= 0 ||
		int64(config.Width)*int64(config.Height) > MaxPixels {
		return nil, ErrImageTooLarge
	}

	img, err := imaging.Decode(bytes.NewReader(data), imaging.AutoOrientation(true))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

// Prepare resizes an image and converts it to a normalized RGB NCHW tensor.
func Prepare(img image.Image, width, height int) ([]float32, error) {
	if img == nil || width <= 0 || height <= 0 {
		return nil, errors.New("invalid preprocessing dimensions")
	}

	resized := imaging.Resize(img, width, height, imaging.Linear)
	pixels := width * height
	tensor := make([]float32, 3*pixels)
	mean := [3]float32{0.485, 0.456, 0.406}
	stddev := [3]float32{0.229, 0.224, 0.225}

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			values := [3]float32{
				float32(r) / 65535,
				float32(g) / 65535,
				float32(b) / 65535,
			}
			i := y*width + x
			for channel := 0; channel < 3; channel++ {
				tensor[channel*pixels+i] = (values[channel] - mean[channel]) / stddev[channel]
			}
		}
	}

	return tensor, nil
}
