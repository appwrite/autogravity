"""Benchmark a running local Docker service; all submitted requests are drained.

Run with host Python (stdlib only). Container must publish 8080 on loopback.
"""
import argparse
import concurrent.futures
import json
import math
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def cgroup(container, path):
    return subprocess.check_output(["docker", "exec", container, "cat", "/sys/fs/cgroup/" + path], text=True)


def usage(container):
    return int(dict(line.split() for line in cgroup(container, "cpu.stat").splitlines())["usage_usec"])


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("container")
    parser.add_argument("variant")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.seconds <= 0:
        parser.error("workers and seconds must be positive")
    port = subprocess.check_output(["docker", "port", args.container, "8080/tcp"], text=True).strip().split(":")[-1]
    url = "http://127.0.0.1:" + port
    for _ in range(100):
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.2)
    else:
        raise RuntimeError("service did not become healthy")
    names = ["dog-portrait.jpg", "panda-bamboo.jpg", "pedestrian-dog.jpg", "bird-branch.jpg"]
    images = [(args.root / "source/internal/testimages/testdata" / name).read_bytes() for name in names]
    # Compare concurrent outputs against this exact Go binary/native library.
    # A Python wheel can use different kernels even at the same ORT version.
    expected = {}

    def request(index):
        name = names[index % len(names)]
        req = urllib.request.Request(url + "/analyze", data=images[index % len(images)], headers={"Content-Type": "image/jpeg"})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - start
        actual = [result["gravity"]["x"], result["gravity"]["y"], result["confidence"]]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in actual):
            raise ValueError(f"Invalid Go HTTP output for {name}: {actual}")
        if name in expected:
            if any(abs(v - ref) > 1e-5 for v, ref in zip(actual, expected[name])):
                raise ValueError(f"Concurrent Go HTTP output changed for {name}: {actual} vs {expected[name]}")
        else:
            expected[name] = actual
        return elapsed

    for i in range(len(names)):
        request(i)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Warm every image and allocate parallel inference workspaces.
        list(pool.map(request, range(max(4, args.workers * 2))))
        barrier = threading.Barrier(args.workers + 1)
        deadline = [0.0]

        def worker(index):
            times, errors = [], []
            barrier.wait()
            while time.perf_counter() < deadline[0]:
                try:
                    times.append(request(index))
                except Exception as error:
                    errors.append(str(error))
                index += 1
            return times, errors

        pending = [pool.submit(worker, i) for i in range(args.workers)]
        cpu_start = usage(args.container)
        start = time.perf_counter()
        deadline[0] = start + args.seconds
        barrier.wait()
        results = [future.result() for future in pending]
        elapsed = time.perf_counter() - start
        cpu_elapsed = (usage(args.container) - cpu_start) / 1e6
    durations = [duration for times, _ in results for duration in times]
    errors = [error for _, failures in results for error in failures]
    peak = int(cgroup(args.container, "memory.peak")) / 2**20
    output = {"variant": args.variant, "workers": args.workers, "successes": len(durations),
              "errors": errors, "elapsed_including_drain": elapsed,
              "requests_per_second": len(durations) / elapsed, "cpu_cores_used": cpu_elapsed / elapsed,
              "peak_cgroup_memory_mib": peak,
              "p50_ms": percentile(durations, .5) * 1000 if durations else None,
              "p95_ms": percentile(durations, .95) * 1000 if durations else None,
              "focal_points": expected}
    print(json.dumps(output), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
