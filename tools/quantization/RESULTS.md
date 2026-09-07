# Native x86 CPU experiment — 2026-09-07

Follow-up with 128 calibration images, 200 public held-out images, and the user's
panda: [expanded evaluation](RESULTS-expanded.md). The decision below records
the initial four-image experiment, before the broader quality assessment.

## Decision

Keep FP32 as the production default. Reduced-range INT8 is promising for speed
and memory, but the four-image held-out comparison shows material focal-point
changes. This calibration dataset is too small to approve production use.

## End-to-end HTTP results

Intel Xeon Platinum 8259CL: **4 physical cores, 8 logical CPUs**, AVX2/AVX-512,
no VNNI. Docker CPU quotas of 1, 2, 4, and 6 were tested; these are logical-CPU
budgets, not dedicated physical-core pinning. Each case used one analysis/client
per CPU, one ONNX intra-op thread, a shared session, and a 4 GiB memory cap.

| CPU budget | FP32 req/s | INT8 req/s | Gain | p95 ms, FP32 / INT8 | Peak MiB, FP32 / INT8 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.85 | 1.18 | +39% | 1211 / 884 | 1133 / 237 |
| 2 | 1.61 | 2.30 | +43% | 1358 / 917 | 1389 / 434 |
| 4 | 3.15 | 4.21 | +34% | 1342 / 1081 | 2030 / 715 |
| 6 | 3.06 | 4.71 | +54% | 2422 / 1535 | 2928 / 789 |

All eight cases completed without request errors or concurrent-output mismatches.
CPU use was approximately 1.00, 2.00, 3.95–3.99, and 5.87–5.92 cores.
Peak memory is **cgroup memory including startup/cache**, not process RSS.

Each case warmed up, measured 30 seconds, and drained every started request;
throughput includes drain time. These are single-run observations, not confidence
intervals or a sustained soak test. FP32 did not benefit from six logical CPUs
on this host; repeat and test affinity before generalizing to other x86 machines.
The workload rotates four JPEGs and includes decode, preprocessing, inference,
and gravity calculation. Latencies are not directly comparable to earlier tests
with a large admission queue.

Raw timings, CPU use, memory, and focal points: [results-x86.json](results-x86.json).

## Quality gate

The same native Go executable/runtime produced both sets of focal points.
Largest absolute held-out coordinate differences:

| Image | Horizontal shift (% image width) | Vertical shift (% image height) |
| --- | --- | --- |
| dog-portrait.jpg | 0.018 | 0.050 |
| panda-bamboo.jpg | 0.018 | 0.177 |
| pedestrian-dog.jpg | 2.556 | 5.885 |
| bird-branch.jpg | 2.126 | 0.544 |

The pedestrian/dog vertical shift is **5.885% of image height**. This and the bird
shift move farther from the existing desired subject regions on known difficult
scenes. Agreement with FP32 is not a substitute for ground-truth crop quality;
neither model's confidence score establishes accuracy.

Both models passed the race-enabled native Go `TestAnalyzeRealModel` and
`TestConcurrentInferenceIsConsistent` checks. The INT8 runs took 29.13 s and
3.93 s, respectively. The existing real-model test does not include the difficult
pedestrian/dog and bird scenes, so passing it does not clear the quality gate.

Calibration used four independent photos, with four different held-out photos.
Next gate: a substantially larger representative calibration/evaluation corpus,
with explicit acceptable focal-point/crop-error thresholds, before another model
selection decision.

## Candidate screening and reproducibility

Initial one-CPU inference-only screening measured FP32 0.835 req/s,
signed QDQ 0.628, unsigned QDQ 0.661, and reduced-range QOperator 1.229.
Only the last candidate advanced to the full HTTP matrix. Both operator
representation and numeric range changed, so the improvement cannot be
attributed to reduced range alone.

The original graph was migrated to opset 13 and preprocessed before static MinMax,
per-channel quantization. The migrated FP32 graph matched original FP32 focal
points on all eight initial Python evaluation inputs.

Initial screening used tensors exported on ARM. Comparing native x86 exports
found 102 of 307,200 elements differed for pedestrian/dog (maximum normalized
difference 0.0175071). The selected INT8 model was therefore **recalibrated using
native x86 tensors** before this final HTTP matrix. Initial Python screening
accuracy files are not the final model's quality results; use the Go outputs
saved above.

- ONNX Runtime: 1.23.2, CPU execution only.
- Quantization dependencies: [requirements.txt](requirements.txt).
- Runtime image: `ghcr.io/appwrite/autogravity:0.0.5`, overridden with the current Go executable.
- Image ID: `sha256:ee9e51803a92417f6fb8957028b15c881947001bc76fd581e7bb681ad1cce5ec`.
- FP32 SHA-256: `8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491`.
- Final INT8 SHA-256: `319a3d3e07f92e5d011ab3639cbb35e4b5a3d05070787789e3e1735fc319a563`.
- Final INT8 model size: 44,211,875 bytes.
- Source archive SHA-256: `c063961d5a431507b4185f0694df2c62b97882e66e40e7297ce3d4bc7e187362` (benchmark scripts were subsequently corrected as documented above).

No production model/default was changed and no existing service was redeployed.
