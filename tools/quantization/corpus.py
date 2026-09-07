"""Prepare a deterministic ECSSD experiment, leaving downloaded data outside git.

Input archives are the authors' images.zip and ground_truth_mask.zip.
Requires Pillow in addition to the quantization environment.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(path):
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    return sum(int(pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))


def extract(archive, destination):
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("unsafe archive path")
        bundle.extractall(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=128)
    parser.add_argument("--evaluation", type=int, default=200)
    args = parser.parse_args()
    if min(args.calibration, args.evaluation) < 1:
        parser.error("both split sizes must be positive")
    root = args.root.resolve()
    extract(root / "images.zip", root / "images")
    extract(root / "masks.zip", root / "masks")
    images = sorted((root / "images").rglob("*.jpg"))
    masks = {p.stem: p for p in (root / "masks").rglob("*.png")}
    if len(images) != 1000 or len(masks) != 1000:
        raise ValueError(f"expected 1000 images/masks, got {len(images)}/{len(masks)}")
    reserved = [root / "user-panda.png"] + [args.fixtures.resolve() / name for name in (
        "rose.png", "portrait.jpg", "puppies.jpg", "person-room.jpg", "dog-portrait.jpg",
        "panda-bamboo.jpg", "pedestrian-dog.jpg", "bird-branch.jpg")]
    seen = [dhash(p) for p in reserved]
    excluded, accepted = [], []
    # Fixed hash ordering, independent of directory listing or image contents.
    for path in sorted(images, key=lambda p: hashlib.sha256(("autogravity-ecssd-v1:" + p.name).encode()).hexdigest()):
        fingerprint = dhash(path)
        if any((fingerprint ^ other).bit_count() <= 4 for other in seen):
            excluded.append(path.name)
            continue
        seen.append(fingerprint)
        accepted.append(path)
    if len(accepted) < args.calibration + args.evaluation:
        raise ValueError("not enough independent images after duplicate filtering")
    rows = []
    for i, path in enumerate(accepted[:args.calibration + args.evaluation]):
        rows.append({"name": "ecssd-" + path.name, "path": str(path),
                     "split": "calibration" if i < args.calibration else "evaluation",
                     "group": "ecssd-" + path.stem, "mask": str(masks[path.stem]), "sha256": sha(path)})
    for path in reserved:
        rows.append({"name": path.name, "path": str(path), "split": "evaluation",
                     "group": "fixture-" + path.stem, "sha256": sha(path)})
    (root / "dataset.json").write_text(json.dumps(rows, indent=2) + "\n")
    provenance = {"source": "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html",
                  "archives": {name: sha(root / name) for name in ("images.zip", "masks.zip")},
                  "calibration": args.calibration, "public_evaluation": args.evaluation,
                  "extra_evaluation": len(reserved), "near_duplicate_filter": "64-bit dHash Hamming <= 4; heuristic, not guaranteed semantic deduplication",
                  "excluded": excluded, "selection": "SHA256(autogravity-ecssd-v1:filename), ascending"}
    (root / "dataset-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
