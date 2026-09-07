"""Compare native Go HTTP focal points and foreground retention on held-out data.

This is an accuracy evaluation, not a throughput benchmark. Crop scoring uses a
maximum-size, boundary-clamped rectangle centered on the reported focal point;
it is a proxy unless the downstream cropper uses the same convention.
"""
import argparse
import concurrent.futures
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def request(url, path):
    content_type = "image/png" if path.suffix == ".png" else "image/jpeg"
    req = urllib.request.Request(url + "/analyze", data=path.read_bytes(), headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.load(response)
    point = [result["gravity"]["x"], result["gravity"]["y"]]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in point):
        raise ValueError(f"invalid focal point: {point}")
    return point


def crop_retention(mask, point, ratio):
    height, width = mask.shape
    crop_width = min(width, height * ratio)
    crop_height = crop_width / ratio
    left = min(max(point[0] * (width - 1) - crop_width / 2, 0), width - crop_width)
    top = min(max(point[1] * (height - 1) - crop_height / 2, 0), height - crop_height)
    x0, y0 = round(left), round(top)
    x1, y1 = round(left + crop_width), round(top + crop_height)
    return float(mask[y0:y1, x0:x1].sum() / mask.sum())


def distribution(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def summarize(rows, variant):
    # Max axis shift is in fractions of image width/height, not pixel distance.
    shifts = [max(abs(a - b) for a, b in zip(row["points"]["fp32"], row["points"][variant])) for row in rows]
    crop_losses = [max(row["retention"]["fp32"][ratio] - row["retention"][variant][ratio]
                       for ratio in row["retention"]["fp32"]) for row in rows]
    return {"images": len(rows), "max_axis_shift": distribution(shifts),
            "shift_above_2pct": sum(v > .02 for v in shifts),
            "shift_above_5pct": sum(v > .05 for v in shifts),
            "shift_above_10pct": sum(v > .10 for v in shifts),
            "worst_aspect_crop_retention_loss": distribution(crop_losses),
            "crop_loss_above_5pp": sum(v > .05 for v in crop_losses),
            "crop_loss_above_10pp": sum(v > .10 for v in crop_losses),
            "mean_retention": {model: {ratio: float(np.mean([r["retention"][model][ratio] for r in rows]))
                                       for ratio in rows[0]["retention"][model]} for model in ("fp32", variant)},
            "largest_shifts": [{"name": row["name"], "shift": shift} for shift, row in
                               sorted(zip(shifts, rows), key=lambda pair: pair[0], reverse=True)[:10]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    items = [item for item in json.loads((args.root / "dataset.json").read_text()) if item["split"] == "evaluation"]
    urls = {"fp32": args.baseline, "int8_4": args.previous, "int8_128": args.candidate}
    for url in urls.values():
        deadline = time.monotonic() + 120
        while True:
            try:
                with urllib.request.urlopen(url + "/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(f"service failed to become healthy: {url}")
            time.sleep(.2)

    def evaluate(item):
        points = {name: request(url, Path(item["path"])) for name, url in urls.items()}
        row = {"name": item["name"], "sha256": item["sha256"], "points": points}
        if "mask" in item:
            with Image.open(item["mask"]) as image:
                mask = np.asarray(image.convert("L")) >= 128
            if not mask.any():
                raise ValueError("empty foreground mask")
            with Image.open(item["path"]) as image:
                if image.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError("image/mask dimensions differ")
            row["retention"] = {name: {label: crop_retention(mask, point, ratio)
                                       for label, ratio in (("1:1", 1), ("4:5", .8), ("16:9", 16 / 9), ("9:16", 9 / 16))}
                                for name, point in points.items()}
        return row

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(evaluate, items):
            rows.append(row)
            if len(rows) % 20 == 0:
                print(json.dumps({"completed": len(rows), "total": len(items)}), flush=True)
    public = [row for row in rows if "retention" in row]
    result = {"public": {variant: summarize(public, variant) for variant in ("int8_4", "int8_128")},
              "extra_images": [row for row in rows if "retention" not in row], "rows": rows,
              "crop_proxy": "largest fitting rectangle, focal-point-centered and boundary-clamped; foreground mask coverage"}
    (args.root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()
