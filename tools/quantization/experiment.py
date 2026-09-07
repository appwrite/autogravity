"""Isolated CPU quantization experiment. Requires pinned ONNX Runtime 1.23.2.

Inputs come from the Go exporter so calibration uses production preprocessing.
The default four photographs are a feasibility sample; external manifests can
provide larger, disjoint calibration and evaluation sets.
"""
import argparse
import concurrent.futures
import json
import resource
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def fixtures(root):
    return json.loads((root / "tensors/manifest.json").read_text())


def tensor(root, item):
    return np.fromfile(root / "tensors" / item["tensor"], dtype="<f4").reshape(1, 3, 320, 320)


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def quantize(root, variant):
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            items = [x for x in fixtures(root) if x["split"] == "calibration"]
            if not items:
                raise ValueError("calibration split is empty")
            self.items = iter(items)
            self.total = len(items)
            self.count = 0

        def get_next(self):
            item = next(self.items, None)
            if item is not None:
                self.count += 1
                if self.count % 16 == 0 or self.count == self.total:
                    print(json.dumps({"calibration_inputs_read": self.count, "total": self.total}), flush=True)
            return None if item is None else {"input.1": tensor(root, item)}

    # Per-channel QDQ uses the axis attribute introduced in ONNX opset 13.
    # Convert operators properly instead of only changing the version number.
    preprocessed = root / "preprocessed-opset13.onnx"
    if not preprocessed.exists():
        original = onnx.load(root / "u2net.onnx")
        version = next(x.version for x in original.opset_import if x.domain == "")
        if version < 13:
            original = onnx.version_converter.convert_version(original, 13)
        onnx.save(original, root / "opset13.onnx")
        quant_pre_process(root / "opset13.onnx", preprocessed, skip_symbolic_shape=True)
    unsigned = variant == "u8u8"
    reduced = variant == "u8s8-reduced"
    quantize_static(
        preprocessed, root / (variant + ".onnx"), Reader(),
        quant_format=QuantFormat.QOperator if reduced else QuantFormat.QDQ,
        activation_type=QuantType.QUInt8 if unsigned or reduced else QuantType.QInt8,
        weight_type=QuantType.QUInt8 if unsigned else QuantType.QInt8,
        per_channel=True,
        reduce_range=reduced,
    )
    print(json.dumps({"variant": variant, "bytes": (root / (variant + ".onnx")).stat().st_size}), flush=True)


def focal_point(output, region):
    x0, y0, x1, y1 = region
    weights = output.reshape(320, 320)[y0:y1, x0:x1].astype(np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0)
    total = weights.sum()
    confidence = min(1.0, float(weights.max()))
    if not total:
        return [0.5, 0.5, confidence]
    x = float((weights.sum(axis=0) * np.arange(x1 - x0)).sum() / total / max(1, x1 - x0 - 1))
    y = float((weights.sum(axis=1) * np.arange(y1 - y0)).sum() / total / max(1, y1 - y0 - 1))
    return [x, y, confidence]


def evaluate(root, variant):
    model = session(root / (variant + ".onnx"))
    rows = []
    for item in fixtures(root):
        output = model.run(["1959"], {"input.1": tensor(root, item)})[0]
        if not np.isfinite(output).all():
            raise ValueError("non-finite saliency output")
        row = {"name": item["name"], "split": item["split"], "point": focal_point(output, item["region"])}
        rows.append(row)
    (root / (variant + "-accuracy.json")).write_text(json.dumps(rows, indent=2))
    print(json.dumps({"variant": variant, "accuracy": rows}), flush=True)


def bench(root, variant, workers, seconds):
    model = session(root / (variant + ".onnx"))
    inputs = [tensor(root, x) for x in fixtures(root) if x["split"] == "evaluation"]
    # Warm every worker concurrently, so the timed window excludes arena growth.
    def warm(_):
        model.run(["1959"], {"input.1": inputs[0]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(warm, range(workers * 2)))
        barrier = threading.Barrier(workers + 1)
        deadline = [0.0]

        def run(worker):
            durations = []
            barrier.wait()
            index = worker
            while time.perf_counter() < deadline[0]:
                start = time.perf_counter()
                output = model.run(["1959"], {"input.1": inputs[index % len(inputs)]})[0]
                elapsed = time.perf_counter() - start
                if not np.isfinite(output).all():
                    raise ValueError("non-finite output during benchmark")
                durations.append(elapsed)
                index += 1
            return durations

        pending = [pool.submit(run, i) for i in range(workers)]
        cpu_start = time.process_time()
        start = time.perf_counter()
        deadline[0] = start + seconds
        barrier.wait()
        durations = [duration for future in pending for duration in future.result()]
        elapsed = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu_start
    result = {
        "variant": variant, "workers": workers, "requests": len(durations),
        "seconds_including_drain": elapsed, "requests_per_second": len(durations) / elapsed,
        "p50_ms": float(np.percentile(durations, 50) * 1000),
        "p95_ms": float(np.percentile(durations, 95) * 1000),
        "cpu_cores_used": cpu_seconds / elapsed,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "runtime": ort.__version__,
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["quantize", "evaluate", "bench"])
    parser.add_argument("variant", choices=["u2net", "s8s8", "u8u8", "u8s8-reduced", "preprocessed-opset13"])
    parser.add_argument("--root", type=Path, default=Path("/work"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if args.workers < 1 or args.seconds <= 0:
        parser.error("workers and seconds must be positive")
    if args.action == "quantize":
        if args.variant not in ("s8s8", "u8u8", "u8s8-reduced"):
            parser.error("choose a quantized candidate")
        quantize(args.root, args.variant)
    elif args.action == "evaluate":
        evaluate(args.root, args.variant)
    else:
        bench(args.root, args.variant, args.workers, args.seconds)
