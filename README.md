# autogravity

`autogravity` is a small Go HTTP service that finds the main visual subject in
an image. It runs U²-NetP with ONNX Runtime and returns the saliency-weighted
centroid as normalized X/Y coordinates. It never crops, stores, or modifies the
submitted image.

## Requirements

- Go 1.25 or newer
- The U²-NetP ONNX model (`make model` downloads and verifies it)
- An ONNX Runtime shared library. Version 1.23.2 is used by the Docker image and
  matches the pinned Go binding.

Download the appropriate ONNX Runtime 1.23.2 archive from the
[official releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2),
extract it, and set `ONNXRUNTIME_LIB` to the full path of its shared library:

```sh
export ONNXRUNTIME_LIB=/absolute/path/to/libonnxruntime.dylib  # macOS
# export ONNXRUNTIME_LIB=/absolute/path/to/libonnxruntime.so   # Linux
```

## Build and run

```sh
make model
make build
./autogravity
```

The server listens on `:8080`. These environment variables are available:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ADDR` | `:8080` | HTTP listen address |
| `MODEL_PATH` | `models/u2netp.onnx` | U²-NetP model path |
| `ONNXRUNTIME_LIB` | required | Full ONNX Runtime shared-library path |

## Performance

On an Apple M3 Pro, image analysis takes about **120 ms per image** with U²-NetP
and CPU-only ONNX Runtime:

| Input | Dimensions | Time per image |
| --- | --- | --- |
| Landscape JPEG | 1280 × 720 | 119.6 ms |
| Portrait PNG | 720 × 1080 | 123.2 ms |

Measured on September 6, 2026, with macOS 26.5.2 (arm64), 18 GiB RAM, Go 1.25.14,
and ONNX Runtime 1.23.2. Each result is the median of five sequential benchmark
samples using `-benchtime=3s` and the checked-in synthetic images.

Timings include decoding, orientation handling, resizing and normalization to
320 × 320, inference, and focal-point calculation. They exclude model startup,
file reads, uploads, and HTTP overhead. Performance varies with hardware and
input images; see [benchmark instructions](CONTRIBUTING.md#benchmarks) to measure
your environment.

## Docker

The image downloads the verified U²-NetP model and the CPU-only ONNX Runtime
library during the build. Docker BuildKit supports both `linux/amd64` and
`linux/arm64`.

```sh
docker build -t autogravity .
docker run --rm -p 8080:8080 autogravity
```

### Container releases

Publishing a GitHub Release with a semantic version tag such as `v1.2.3`
builds and pushes a multi-architecture image to GitHub Container Registry:

```text
ghcr.io/appwrite/autogravity:1.2.3
ghcr.io/appwrite/autogravity:1.2
ghcr.io/appwrite/autogravity:1
ghcr.io/appwrite/autogravity:latest
```

Prereleases receive only their full version tag and do not update `latest`.
Published images include build provenance and an SBOM.

## API

Health check (returns 503 until the model is loaded):

```sh
curl -sS http://localhost:8080/healthz
```

```json
{
  "status": "ok"
}
```

Send a JPEG, PNG, or WebP image as a multipart `image` field:

```sh
curl -sS -X POST http://localhost:8080/analyze \
  -F 'image=@photo.jpg'
```

Or send the image as the raw request body:

```sh
curl -sS -X POST http://localhost:8080/analyze \
  -H 'Content-Type: image/jpeg' \
  --data-binary '@photo.jpg'
```

Example response:

```json
{
  "gravity": {
    "x": 0.68,
    "y": 0.37
  },
  "confidence": 0.91
}
```

Coordinates are in `[0.0, 1.0]`, measured from the oriented image's top-left
corner. EXIF orientation is applied before analysis. Images are fitted within
the model's 320x320 input using neutral padding, without stretching or
cropping. Padding is excluded from the focal-point calculation. Confidence is
the peak activation in the model's fused saliency map, clamped to `[0.0, 1.0]`.

Requests are limited to 10 MiB and decoded images to 20 megapixels. Separate
upload and analysis admission limits bound buffered-body and decoded-image
memory without allowing slow uploads to reserve inference capacity. The model
is loaded once at startup and its shared inference session is reused safely
across requests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, quality
evaluation, benchmark reproduction, and the project layout.
