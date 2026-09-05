package imageutil

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"testing"
)

func TestDecodeWebP(t *testing.T) {
	data, err := base64.StdEncoding.DecodeString(
		"UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA",
	)
	if err != nil {
		t.Fatal(err)
	}

	img, err := Decode(data)
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if got := img.Bounds().Size(); got != (image.Point{X: 1, Y: 1}) {
		t.Fatalf("decoded size = %v, want 1x1", got)
	}
}

func TestDecodeAppliesEXIFOrientation(t *testing.T) {
	source := image.NewRGBA(image.Rect(0, 0, 2, 1))
	source.Set(0, 0, color.RGBA{R: 255, A: 255})
	source.Set(1, 0, color.RGBA{B: 255, A: 255})
	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, source, nil); err != nil {
		t.Fatal(err)
	}

	img, err := Decode(withEXIFOrientation(encoded.Bytes(), 6))
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if got := img.Bounds().Size(); got != (image.Point{X: 1, Y: 2}) {
		t.Fatalf("oriented size = %v, want 1x2", got)
	}
}

func TestDecodeRejectsUnknownFormat(t *testing.T) {
	_, err := Decode([]byte("not an image"))
	if !errors.Is(err, ErrUnsupportedFormat) {
		t.Fatalf("Decode() error = %v, want ErrUnsupportedFormat", err)
	}
}

func TestDecodeRejectsExcessivePixelCount(t *testing.T) {
	_, err := Decode(oversizedPNGHeader(5_000, 5_000))
	if !errors.Is(err, ErrImageTooLarge) {
		t.Fatalf("Decode() error = %v, want ErrImageTooLarge", err)
	}
}

func TestPrepareProducesNormalizedNCHW(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 128, B: 0, A: 255})

	got, err := Prepare(img, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("tensor length = %d, want 3", len(got))
	}
	want := []float32{
		(1 - 0.485) / 0.229,
		(float32(128)/255 - 0.456) / 0.224,
		(0 - 0.406) / 0.225,
	}
	for i := range want {
		if delta := got[i] - want[i]; delta < -0.0001 || delta > 0.0001 {
			t.Fatalf("tensor[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func withEXIFOrientation(jpegData []byte, orientation uint16) []byte {
	tiff := []byte{
		'I', 'I', 0x2a, 0x00,
		0x08, 0x00, 0x00, 0x00,
		0x01, 0x00,
		0x12, 0x01,
		0x03, 0x00,
		0x01, 0x00, 0x00, 0x00,
		byte(orientation), byte(orientation >> 8), 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
	}
	payload := append([]byte("Exif\x00\x00"), tiff...)
	segment := []byte{0xff, 0xe1, byte((len(payload) + 2) >> 8), byte(len(payload) + 2)}
	segment = append(segment, payload...)
	result := append([]byte{}, jpegData[:2]...)
	result = append(result, segment...)
	return append(result, jpegData[2:]...)
}

func oversizedPNGHeader(width, height uint32) []byte {
	result := append([]byte{}, []byte("\x89PNG\r\n\x1a\n")...)
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:4], width)
	binary.BigEndian.PutUint32(ihdr[4:8], height)
	ihdr[8] = 8 // Bit depth.
	ihdr[9] = 2 // Truecolor.
	chunkType := []byte("IHDR")
	length := make([]byte, 4)
	binary.BigEndian.PutUint32(length, uint32(len(ihdr)))
	result = append(result, length...)
	result = append(result, chunkType...)
	result = append(result, ihdr...)
	checksum := crc32.ChecksumIEEE(append(chunkType, ihdr...))
	crc := make([]byte, 4)
	binary.BigEndian.PutUint32(crc, checksum)
	return append(result, crc...)
}
