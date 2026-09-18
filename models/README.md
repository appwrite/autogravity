# Model artifacts

The default, `u2net-int8.onnx`, is included in this repository (44,211,876
bytes) so clean builds need no private host or mutable experimental download.
Its SHA-256 is verified by `make model` and Docker builds:

```text
b340186f56660b6665e494aab912e5f8e9adbc2317181c77fd01aa226f06553b
```

This is a modified [U²-Net](https://github.com/xuebinqin/U-2-Net) model, whose
upstream Apache-2.0 license is included in `U2NET_LICENSE`. The original FP32 ONNX
artifact is obtained from the rembg release URL pinned in the Dockerfile and
Makefile (SHA-256 `8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491`).

Modifications made on 2026-09-07: migrate the graph to opset 13, preprocess it,
and apply static MinMax, per-channel, reduced-range U8/S8 QOperator quantization
with ONNX Runtime 1.23.2. Calibration uses 128 ECSSD images with native x86 Go
preprocessing. The user-provided panda and all evaluation images were excluded
from calibration. Evaluation used 200 separate ECSSD images plus nine extra
fixtures. Median focal-point shift versus FP32 was 0.10% of an image dimension,
p95 was 0.91%, and the maximum was 7.4%. At four CPUs on the tested x86 Xeon,
two HTTP runs measured 38–45% more throughput and 64–65% less peak cgroup memory.
Raw calibration and evaluation images are not bundled with the model.

Both architectures default to `MODEL_PRECISION=int8`. Set `MODEL_PRECISION=fp32`
to use the bundled FP32 model. An explicit `MODEL_PATH` overrides precision.
There is no automatic fallback on model errors or architecture-based selection.
The reported throughput gains were measured on x86-64, not ARM64.

## Face detection

`face_detection_yunet_2023mar.onnx` is the fixed 640x640
[YuNet face detector](https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet)
from OpenCV Zoo, pinned to upstream commit
`f12e12798e8314f7c074a6656816c048dcc95b7a`. The checked-in artifact is 232,589
bytes with SHA-256:

```text
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```

The artifact is redistributed under the MIT license in `YUNET_LICENSE`. Images
are aspect-fitted into its input with black padding and no stretching. YuNet
runs before U²-Net; a face at or above `FACE_SCORE_THRESHOLD` supplies the
focal point, while images without a reliable face retain the saliency result.
The 0.85 default retains the licensed clear-face fixture while rejecting a
0.81 false positive on the two-puppy regression image. Deliberately blurred or
obscured faces may not reach the threshold.

## FocalNet

FocalNet is Appwrite's own model, used when `MODEL_BACKEND=focalnet`. The
weights are `focalnet-human.onnx` from the public
[FocalNet `2026-09-14-rc1` release](https://github.com/appwrite/focalnet/releases/tag/2026-09-14-rc1)
(20,398,992 bytes). `make model-focalnet` downloads it, verifies this SHA-256,
and stages a copy at `models/optional/focalnet-human.onnx` for Docker:

```text
59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439
```

The graph is format v2: inputs `image` `[1,3,256,256]`, `boxes` `[1,128,4]`,
`content` `[1,4]`; outputs `importance` `[1,1,64,64]` and `crop_scores`
`[1,128]`. Autogravity generates FocalNet's candidate crops, applies the 0.05
importance-retention gate, and returns the selected crop center. A
contract-compatible placeholder lives at `internal/focalnet/testdata/dummy.onnx`
for Go tests; it is image-independent and must not be copied into
`models/focalnet-human.onnx` or a production image. Docker accepts only the
published checksum above. The artifact is redistributed under the MIT license
in `FOCALNET_LICENSE`. See [appwrite/focalnet](https://github.com/appwrite/focalnet)
for training code and results.

PR image builds still start without the file. `MODEL_BACKEND=focalnet` fails at
startup if it is missing. Release images require the real checksum
(`REQUIRE_FOCALNET_MODEL=1`). Run `make model-focalnet` before `docker build`,
or set repo secret `FOCALNET_GITHUB_TOKEN` if a workflow cannot download the
release.
