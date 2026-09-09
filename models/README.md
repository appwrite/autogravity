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
