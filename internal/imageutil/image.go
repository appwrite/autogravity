// Package imageutil decodes and prepares images for the saliency model.
package imageutil

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

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

// Prepare fits an image within the requested dimensions without changing its
// aspect ratio. It returns a normalized RGB NCHW tensor and the rectangle
// occupied by the image; unused tensor pixels are neutral zero padding.
func Prepare(img image.Image, width, height int) ([]float32, image.Rectangle, error) {
	if img == nil || width <= 0 || height <= 0 {
		return nil, image.Rectangle{}, errors.New("invalid preprocessing dimensions")
	}

	sourceWidth, sourceHeight := img.Bounds().Dx(), img.Bounds().Dy()
	if sourceWidth <= 0 || sourceHeight <= 0 {
		return nil, image.Rectangle{}, errors.New("invalid image dimensions")
	}

	scale := math.Min(float64(width)/float64(sourceWidth), float64(height)/float64(sourceHeight))
	resizedWidth := max(1, min(width, int(math.Round(float64(sourceWidth)*scale))))
	resizedHeight := max(1, min(height, int(math.Round(float64(sourceHeight)*scale))))
	offsetX := (width - resizedWidth) / 2
	offsetY := (height - resizedHeight) / 2
	content := image.Rect(offsetX, offsetY, offsetX+resizedWidth, offsetY+resizedHeight)
	resized := imaging.Resize(img, resizedWidth, resizedHeight, imaging.Linear)

	pixels := width * height
	tensor := make([]float32, 3*pixels)
	mean := [3]float32{0.485, 0.456, 0.406}
	stddev := [3]float32{0.229, 0.224, 0.225}

	for y := 0; y < resizedHeight; y++ {
		for x := 0; x < resizedWidth; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			values := [3]float32{
				float32(r) / 65535,
				float32(g) / 65535,
				float32(b) / 65535,
			}
			i := (y+offsetY)*width + x + offsetX
			for channel := 0; channel < 3; channel++ {
				tensor[channel*pixels+i] = (values[channel] - mean[channel]) / stddev[channel]
			}
		}
	}

	return tensor, content, nil
}
