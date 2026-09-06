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

Run tests with:

```sh
make test
```

The default suite uses checked-in photographs in JPEG, PNG, and lossy,
lossless, and transparent WebP formats. It checks decoding, all eight EXIF
orientations, normalization, letterbox padding, raw and multipart uploads,
corrupt images, and recovery after analysis failures. Fixtures and their
source license live in `internal/testimages/testdata`.

To also run the real U²-NetP model through the HTTP handler:

```sh
export ONNXRUNTIME_LIB=/absolute/path/to/libonnxruntime.dylib
make test-integration
```

This downloads and verifies the model, then runs race-enabled tests including
subject-location checks, mirrored-image consistency, and equivalent raw and
multipart results. The integration suite requires a working runtime and model;
it fails rather than silently skipping when they are missing. GitHub Actions
installs the pinned runtime and runs this suite on every pull request and push
to `main`. Default tests need neither the runtime nor network access.

### mise

If you use [mise](https://mise.jdx.dev/), the repository pins Go and exposes
the common development tasks:

```sh
mise install
mise run ci       # formatting, vet, race-enabled tests, and build
mise run model    # download and verify U²-NetP
```

GitHub Actions runs `mise run ci` and validates the Docker image for both
`linux/amd64` and `linux/arm64` on pull requests and pushes to `main`.

### Benchmark

An end-to-end benchmark uses the included landscape JPEG and portrait PNG. It
measures image decoding, orientation handling, resize and normalization, ONNX
inference, and focal-point calculation. Model startup is excluded.

```sh
export ONNXRUNTIME_LIB=/absolute/path/to/libonnxruntime.dylib
mise run bench
```

Use `.so` instead of `.dylib` on Linux. Regenerate the synthetic example
images with `go generate ./internal/benchmark`.

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

## Layout

```text
cmd/autogravity/       HTTP server and lifecycle
internal/imageutil/    decoding, EXIF orientation, resize, normalization
internal/saliency/     ONNX Runtime model session and inference
internal/gravity/      saliency-weighted focal-point calculation
models/                local model location (ONNX files are gitignored)
```
