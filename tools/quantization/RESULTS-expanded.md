# Expanded x86 INT8 evaluation — 2026-09-07

This report records the experiment before the default was changed. The service
now defaults to `MODEL_PRECISION=int8` on all architectures, with `fp32` and
custom-path overrides. The measured throughput gains below remain x86-specific.

## Outcome

The larger held-out sample supports INT8 as a useful speed/quality trade-off when
small regressions are acceptable. It does not establish lossless conversion.
The 128-image candidate has a median focal-point shift of 0.10% and p95 of 0.91%
of an image dimension; occasional larger shifts remain. Production was not
redeployed and FP32 remains the configured default.

## Dataset and separation

Downloaded the authors' [ECSSD images and foreground masks](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html).
From 1,000 images, deterministic filename-hash ordering selected 128 calibration
images and 200 separate public evaluation images. Nine additional evaluation
images comprise the user's panda and eight repository photos. The user image
was never used for calibration.

Exact-byte validation and a 64-bit dHash near-duplicate heuristic found no
overlap. This does not guarantee semantic independence or establish anything
about the pretrained model's training history. The public sample is not a
substitute for representative production traffic. Data remains outside git;
[dataset-expanded.json](dataset-expanded.json) records selection, archive
checksums, and individual image hashes.

The previous four-image candidate is compared on the same 200 public held-out
images. Four of the nine extra fixtures were its calibration inputs, so do not
compare aggregate extra-fixture accuracy as if all nine were held out for it.

## Quality measurements

All three models used the same compiled Go HTTP service and native ONNX Runtime
1.23.2. The 209 paired cases completed successfully: 627 model/image requests.
Every raw focal point and per-aspect crop score is retained in
[results-expanded.json](results-expanded.json).

| Metric, 200 public images | INT8 / 4 calibration images | INT8 / 128 calibration images |
| --- | ---: | ---: |
| Median max-axis shift | 0.096% | 0.101% |
| p95 max-axis shift | 1.011% | 0.911% |
| Maximum max-axis shift | 8.406% | 7.433% |
| Images with shift above 5% | 1/200 | 2/200 |
| p95 worst-aspect crop retention loss | 0.527 pp | 0.666 pp |
| Maximum worst-aspect crop retention loss | 9.491 pp | 5.997 pp |
| Images losing more than 5 pp in any crop shape | 2/200 | 1/200 |
| Images losing more than 10 pp in any crop shape | 0/200 | 0/200 |

Max-axis shift is max(abs(INT8.x − FP32.x), abs(INT8.y − FP32.y)).
Horizontal and vertical shifts are normalized by their respective image
dimensions, not a common pixel-distance scale.

Crop retention measures the fraction of the annotated foreground mask retained
by the largest fitting rectangle of aspect 1:1, 4:5, 16:9, or 9:16, centered on
the reported focal point and clamped to image boundaries. Each image's reported
loss is the largest FP32-to-INT8 reduction among those four shapes; pp means
percentage points of foreground coverage. This is a proxy, not an evaluation of
composition, face visibility, or an unspecified downstream cropper.

Mean foreground retention is essentially unchanged (slightly higher for INT8
in this sample). The worst new-candidate crop loss is 5.997 pp on the deer image,
ECSSD 0391. The largest focal-point shift is on ECSSD 0249, a different image.
The larger calibration set is not uniformly better: its p95 crop loss is slightly
higher, while its maximum crop loss is lower. Do not infer a guaranteed quality
improvement from calibration count alone.

### User's panda and difficult fixtures

On the supplied 976×549 panda image:

| Model | x | y |
| --- | ---: | ---: |
| FP32 | 0.655951 | 0.506859 |
| INT8 / 4 images | 0.655702 | 0.504145 |
| INT8 / 128 images | 0.656639 | 0.505224 |

The expanded candidate moves about 0.7 pixels horizontally and 0.9 pixels
vertically relative to FP32. No ground-truth mask was supplied for this image;
this measures agreement, not an independent judgment of its ideal focal point.

The pedestrian/dog vertical difference remains 5.867% of image height; the bird's
horizontal difference is 2.175% of width. Broader calibration did not fix these
known difficult cases.

## Repeated end-to-end timing

Two sequential FP32/INT8 pairs used four CPUs, four clients/analysis slots, one
ONNX intra-op thread, and a 4 GiB memory limit. Each fresh container was warmed,
then measured for 30 seconds and drained completely. This uses the same four
JPEGs as the first experiment, not the public corpus, to keep timings comparable.
Accuracy evaluation and race tests did not overlap these timing runs.

| Repeat | FP32 req/s | INT8 req/s | Gain | p95 ms, FP32 / INT8 | Peak MiB, FP32 / INT8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.954 | 4.063 | +37.5% | 1485 / 1108 | 2019 / 712 |
| 2 | 2.982 | 4.312 | +44.6% | 1432 / 1010 | 2027 / 726 |

All four cases completed without request errors or concurrent-output mismatches.
The new model gains **37.5–44.6% throughput**, reduces peak cgroup memory by
**64.2–64.7%**, and lowers p95 latency in both runs. Memory includes startup/cache
and is not process RSS. Two short runs do not establish a sustained-load SLO
or a statistically precise speedup. These new timings test four CPUs only; do
not transplant the previous candidate's six-CPU result to this model.

Raw results: [timings-expanded.json](timings-expanded.json).

Given the stated tolerance for minor regressions, this candidate is reasonable
for an **opt-in x86 rollout**, with FP32 retained as a fallback. A tested starting
profile is four CPU units, four analysis slots, one intra-op thread, and the
existing 4 GiB cap. Lower memory limits and the downstream cropper's exact
behavior need separate verification. This experiment does not deploy that profile.

## Reproducibility

Same Intel Xeon Platinum 8259CL as the first experiment: 4 physical cores,
8 logical CPUs, CPU execution only, no VNNI. The expanded model uses static
MinMax, per-channel, reduced-range unsigned-activation/signed-weight QOperator
quantization of the opset-13 preprocessed graph. Calibration was capped at two
CPUs and 8 GiB; tensors were exported natively on x86 using production Go
preprocessing. Evaluation used four total clients across three shared-session
Go services; its duration is not a throughput benchmark.

- Expanded INT8 SHA-256: `b340186f56660b6665e494aab912e5f8e9adbc2317181c77fd01aa226f06553b`.
- Expanded INT8 size: 44,211,876 bytes.
- Go executable SHA-256: `10798e92bea590deb0475dd66fffca071cb3b56e78d32f6f2c469dea2cb684fd`.
- Native exporter SHA-256: `98f37c00db2d7d9f607f05f176b3700ceab2c175a7948e0a378b82c7037e3187`.
- Original/previous model and runtime image hashes: [first experiment](RESULTS.md).
- Commands and pinned dependencies: [README.md](README.md).

The added crop-scoring tests and dataset-exporter tests pass. The expanded model
also passed native race-enabled `TestAnalyzeRealModel` (29.73 s) and
`TestConcurrentInferenceIsConsistent` (3.90 s). Go's race detector does not prove
the absence of races inside the native runtime; HTTP consistency checks provide
additional observed evidence, not a formal thread-safety guarantee. The
experiment does not modify or redeploy existing services.
