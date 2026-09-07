# CPU quantization experiment

These tools create experimental models and measure their accuracy and speed.
They do not replace the production model or change server defaults.

Measured native x86 findings: [RESULTS.md](RESULTS.md).
Expanded public-dataset and user-image findings: [RESULTS-expanded.md](RESULTS-expanded.md).

## Calibration and evaluation

Run the exporter on the target architecture, from the repository root. It uses
the production Go decoder, letterboxing, alpha handling, and normalization.
Small differences in resizing between ARM and x86 can change input tensors.

```sh
go run ./tools/quantization/export -out /absolute/experiment/tensors
cp models/u2net.onnx /absolute/experiment/u2net.onnx
python3 -m venv /absolute/experiment/venv
/absolute/experiment/venv/bin/pip install -r tools/quantization/requirements.txt
```

Four independent photographs are used for calibration: rose, portrait, puppies,
and person in a room. Four different photographs are held out: dog portrait,
panda, pedestrian with dog, and bird on a branch. This tiny sample is for
feasibility testing only, not sufficient to approve a production model. The
pedestrian/dog and bird scenes are known weaknesses of the existing FP32 model.

For a larger corpus, pass `-manifest /absolute/dataset.json` to the Go exporter.
Each entry specifies an image path relative to the manifest, a unique basename,
an explicit split, and a source-photo group:

```json
[
  {"name":"photo-a.jpg","path":"images/a.jpg","split":"calibration","group":"source-a"},
  {"name":"photo-b.jpg","path":"images/b.jpg","split":"evaluation","group":"source-b"}
]
```

The exporter rejects duplicate file contents and groups that cross splits, and
records image SHA-256 hashes in its output. Assign re-encodings, crops, and resizes
of one photo the same group; byte hashes alone cannot identify those variants.
Both splits are required. Run the exporter on x86 for the x86 experiment.
The Python calibration/evaluation commands consume this expanded manifest;
the HTTP benchmark still uses its fixed four-image workload.

Small focal-point regressions are acceptable when accompanied by substantial
performance gains. Evaluate the median, p95, and maximum coordinate shifts and
inspect changed crops, rather than requiring exact FP32 agreement. The existing
four-image result is insufficient to establish the frequency of larger shifts.

```sh
/absolute/experiment/venv/bin/python tools/quantization/experiment.py quantize s8s8 --root /absolute/experiment
/absolute/experiment/venv/bin/python tools/quantization/experiment.py quantize u8u8 --root /absolute/experiment
/absolute/experiment/venv/bin/python tools/quantization/experiment.py quantize u8s8-reduced --root /absolute/experiment
/absolute/experiment/venv/bin/python tools/quantization/experiment.py evaluate u2net --root /absolute/experiment
/absolute/experiment/venv/bin/python tools/quantization/experiment.py evaluate u8s8-reduced --root /absolute/experiment
```

The conversion first migrates the graph to opset 13, required by per-channel
QDQ, and runs ONNX preprocessing. Compare `evaluate preprocessed-opset13` against
`evaluate u2net` to isolate changes from this migration. All candidates use
static MinMax calibration and per-channel weight quantization:

| Candidate | Representation | Activations | Weights |
| --- | --- | --- | --- |
| `s8s8` | QDQ | signed INT8 | signed INT8 |
| `u8u8` | QDQ | unsigned INT8 | unsigned INT8 |
| `u8s8-reduced` | QOperator | unsigned INT8 | signed, reduced-range |

The reduced-range experiment changes both the operator representation and the
weight range. Its results cannot be attributed to either change independently.

## Measurements

`experiment.py bench MODEL --root DIR --workers N --seconds 30` measures native
inference on four held-out, preprocessed tensors. Use an external CPU limit
matching N. Each process loads one shared CPU session with one intra-op thread.
The measurement drains all submitted work and includes drain time in throughput.
It excludes image decoding, preprocessing, HTTP, model startup, and warmup.

`http_bench.py` measures the actual Go HTTP service on the original JPEGs,
including decoding, preprocessing, inference, and gravity calculation. It uses
one HTTP client per analysis slot, warms the service, then drains every started
request. This measures capacity without a large client-side admission queue.
Every response is checked against serial responses from the same Go service;
reported `focal_points` permit direct FP32/INT8 accuracy comparisons. Python
reference outputs should not be used as bit-exact references for another runtime
build or preprocessing architecture.

On the remote Linux host, prepare a workspace containing:

- `source/`: this repository's source and photographs;
- `autogravity`: the current Linux CGO executable built from that source;
- `u2net.onnx` and the experimental model;
- `http_bench.py` and `run-http.sh` from this directory.

Then run:

```sh
sh /absolute/experiment/run-http.sh /absolute/experiment u8s8-reduced 30
```

The runner compares 1, 2, 4, and 6 logical CPUs, with a 4 GiB memory limit, one
analysis per CPU, and one ONNX intra-op thread. It mounts the workspace read-only
into the service and publishes only a random loopback port. The existing
`ghcr.io/appwrite/autogravity:0.0.5` image supplies ONNX Runtime 1.23.2; its
executable is overridden with the newly built workspace binary. Results are
written to `http-MODEL-CPUS.json`. Containers are stopped after each case.

CPU usage is measured from cgroup CPU-time deltas. Peak cgroup memory includes
startup and may include file cache; it is not the same metric as process RSS.
Repeat measurements on an otherwise quiet host before making a deployment
decision. On machines with SMT, logical CPUs are not physical cores.

## Expanded public evaluation

`corpus.py` prepares a deterministic 128/200 calibration/evaluation split of
[ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), with
eight existing photos and a user-supplied `user-panda.png` as extra held-out cases.
Download the authors' `data/ECSSD/images.zip` and
`data/ECSSD/ground_truth_mask.zip` into an isolated workspace as `images.zip` and
`masks.zip`. Keep the data outside git; public availability does not establish
permission to redistribute every original photograph.

```sh
python corpus.py --root /expanded --fixtures /work/source/internal/testimages/testdata
./export -manifest /expanded/dataset.json -out /expanded/tensors
python experiment.py quantize u8s8-reduced --root /expanded
```

These paths assume the original workspace is mounted at `/work` and the expanded
one at `/expanded`; use the pinned venv Python and native x86 exporter. The expanded
workspace also needs the original `u2net.onnx` and `preprocessed-opset13.onnx`
(symlinks into the mounted original workspace work). Install `requirements.txt`
in the original venv. Place `compare_http.py` in the expanded workspace, then on
the Docker host run:

```sh
sh tools/quantization/run-compare.sh /absolute/original-workspace /absolute/expanded-workspace
```

The runner starts three isolated Go services: FP32, the original four-image INT8,
and the expanded 128-image INT8. Four clients compare all 209 held-out images.
It writes `comparison.json`, including every focal point and public-mask crop
retention, then removes its containers. This is accuracy evaluation, not a
throughput measurement. Mask coverage is a crop-quality proxy, not a human
judgment of composition, and the crop geometry must match the downstream
cropper before treating these numbers as production crop scores.

Selection is fixed by filename hashes before inference. A 64-bit dHash filter
excludes approximate duplicates within Hamming distance four, including against
reserved fixtures; this is a heuristic, not a guarantee against semantic overlap.
The exported manifest additionally rejects exact duplicate bytes. The split is
held out from INT8 calibration, not a claim about the pretrained model's training
history. Archive checksums and selected image hashes are recorded for auditing.

`run-http.sh DIR CANDIDATE 30 '4'` restricts the timing comparison to four CPUs;
omitting the fourth argument retains the 1/2/4/6 matrix. Timing still uses the
original four-image HTTP workload so it is comparable to the first experiment.
