//go:build ignore

package main

import (
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"log"
	"math"
	"os"
)

func main() {
	landscape := gradient(1280, 720, color.RGBA{24, 50, 76, 255}, color.RGBA{119, 174, 184, 255})
	fillEllipse(landscape, 900, 345, 175, 245, color.RGBA{239, 154, 74, 255})
	fillEllipse(landscape, 900, 205, 92, 92, color.RGBA{252, 218, 171, 255})
	fillEllipse(landscape, 865, 188, 12, 12, color.RGBA{36, 43, 55, 255})
	fillEllipse(landscape, 935, 188, 12, 12, color.RGBA{36, 43, 55, 255})
	writeJPEG("testdata/landscape.jpg", landscape)

	portrait := gradient(720, 1080, color.RGBA{229, 218, 200, 255}, color.RGBA{142, 176, 164, 255})
	fillEllipse(portrait, 315, 620, 185, 315, color.RGBA{129, 63, 79, 255})
	fillEllipse(portrait, 315, 320, 130, 150, color.RGBA{232, 184, 147, 255})
	fillEllipse(portrait, 270, 300, 13, 13, color.RGBA{42, 38, 43, 255})
	fillEllipse(portrait, 360, 300, 13, 13, color.RGBA{42, 38, 43, 255})
	writePNG("testdata/portrait.png", portrait)
}

func gradient(width, height int, top, bottom color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		t := float64(y) / float64(height-1)
		c := color.RGBA{
			R: interpolate(top.R, bottom.R, t),
			G: interpolate(top.G, bottom.G, t),
			B: interpolate(top.B, bottom.B, t),
			A: 255,
		}
		for x := 0; x < width; x++ {
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

func fillEllipse(img *image.RGBA, centerX, centerY, radiusX, radiusY int, c color.RGBA) {
	for y := centerY - radiusY; y <= centerY+radiusY; y++ {
		for x := centerX - radiusX; x <= centerX+radiusX; x++ {
			dx := float64(x-centerX) / float64(radiusX)
			dy := float64(y-centerY) / float64(radiusY)
			if dx*dx+dy*dy <= 1 {
				img.SetRGBA(x, y, c)
			}
		}
	}
}

func interpolate(a, b uint8, t float64) uint8 {
	return uint8(math.Round(float64(a)*(1-t) + float64(b)*t))
}

func writeJPEG(name string, img image.Image) {
	file, err := os.Create(name)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	if err := jpeg.Encode(file, img, &jpeg.Options{Quality: 88}); err != nil {
		log.Fatal(err)
	}
}

func writePNG(name string, img image.Image) {
	file, err := os.Create(name)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	encoder := png.Encoder{CompressionLevel: png.BestSpeed}
	if err := encoder.Encode(file, img); err != nil {
		log.Fatal(err)
	}
}
