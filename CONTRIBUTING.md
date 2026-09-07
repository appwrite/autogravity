# Contributing

See the [README](README.md#requirements) for prerequisites and instructions to
build and run the service.

## Development setup

If you use [mise](https://mise.jdx.dev/), the repository pins Go and exposes
the common development tasks:

```sh
mise install
mise run ci       # formatting, vet, race-enabled tests, and build
mise run model    # download and verify U²-Net
```

GitHub Actions runs `mise run ci` and validates the Docker image for both
`linux/amd64` and `linux/arm64` on pull requests and pushes to `main`.

## Tests

Run tests with:

```sh
make test
```

The default suite uses checked-in photographs in JPEG, PNG, and lossy,
lossless, and transparent WebP formats. It checks decoding, all eight EXIF
orientations, normalization, letterbox padding, raw and multipart uploads,
corrupt images, and recovery after analysis failures. Fixtures and their
source licenses live in [the fixture gallery](internal/testimages/testdata/README.md).
Additional natural photographs cover a dog low in a portrait and two puppies in
grass. A licensed panda eating bamboo is also included as a regression for a
previously reported failure with a similar image.

To also run the real U²-Net model through the HTTP handler:

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

Integration tests follow `MODEL_PRECISION` (default `int8`) and honor explicit
`MODEL_PATH` overrides. `make test-integration-fp32` tests the FP32 environment
switch; CI runs both modes. The INT8 binary is intentionally versioned in
`models/` and checksum-verified, so release builds do not need the experimental
SSH host or the calibration dataset. See `models/README.md` for provenance.

The gallery also includes three difficult natural scenes with manually annotated
subject regions. Run `make evaluate` to check a person in a room, a pedestrian
with a dog, and a bird above tree foliage. Full U²-Net passes the person-in-room
case, but the other two still fail. This quality evaluation is separate from the
CI regression gate; its expected regions are not adjusted to accept incorrect
model predictions.

## Benchmarks

To measure preprocessing time and Go allocations without loading ONNX Runtime:

```sh
go test ./internal/imageutil -run '^$' -bench BenchmarkPrepare -benchmem -count=3
```

`BenchmarkPrepareInto` measures the reusable input-buffer path used by the HTTP
handler. `BenchmarkPrepare` includes input-buffer allocation. The already-sized
case isolates normalization from resizing. These are preprocessing measurements,
not end-to-end inference throughput.

The benchmark uses the included landscape JPEG and portrait PNG. It measures image decoding, orientation handling, resize and normalization, ONNX
inference, and focal-point calculation. Model startup is excluded.

```sh
export ONNXRUNTIME_LIB=/absolute/path/to/libonnxruntime.dylib
mise run bench
```

Use `.so` instead of `.dylib` on Linux. Regenerate the synthetic example
images with `go generate ./internal/benchmark`.

To reproduce the [README benchmark results](README.md#performance), verify the
model and collect five samples of at least three seconds per image:

```sh
make model
go test -run '^$' -bench '^BenchmarkAnalyze$' -benchmem -benchtime=3s -count=5 ./internal/benchmark
```

Set `ONNXRUNTIME_LIB` before running either benchmark command; without it, the
benchmark is skipped. Report the median `ns/op` across the five samples,
converted to milliseconds, together with the hardware, OS, Go version, and
ONNX Runtime version. The fixtures are synthetic, and the benchmark runs
analyses sequentially. File reads, HTTP handling, uploads, and model startup
are excluded.

To tune production throughput, benchmark `MAX_CONCURRENT_ANALYSES` values such
as 1, 2, 4, and 8 under an HTTP workload while recording requests/second,
latency percentiles, and peak memory. Test `ONNX_INTRA_OP_THREADS` alongside it:
the default of 1 is intended for concurrent requests, while a higher value may
reduce single-request latency when more CPU cores are available.

## Layout

```text
cmd/autogravity/       HTTP server and lifecycle
internal/imageutil/    decoding, EXIF orientation, resize, normalization
internal/saliency/     ONNX Runtime model session and inference
internal/gravity/      saliency-weighted focal-point calculation
models/                local model location (ONNX files are gitignored)
```
