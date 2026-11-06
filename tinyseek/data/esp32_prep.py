"""Cull ESP32 capture sessions: drop blurry, dark and near-duplicate photos.

    python -m tinyseek.data.esp32_prep data/esp32cam/raw --out data/esp32cam/manifest.json

Per photo it measures sharpness (variance of the Laplacian, relative to the session's
median so a dim session is not penalized as a whole), brightness, and a 256-bit
difference hash. A photo is dropped when it is much blurrier than its session (motion,
a hand over the lens), nearly black, or a near-duplicate of the last kept photo in the
same session (the rig sat still between interval shots), or listed in --exclude after a
look at the contact sheet (a hand over the lens is sharp enough to pass). The manifest lists every photo
with its metrics and the reason it was dropped, so the choice can be reviewed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def sharpness(gray: Image.Image) -> float:
    lap = np.asarray(gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128)),
                     dtype=np.float32)
    return float(lap.var())


def dhash(gray: Image.Image, size: int = 16) -> np.ndarray:
    small = np.asarray(gray.resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    return (small[:, 1:] > small[:, :-1]).flatten()


def measure(path: Path) -> dict:
    gray = Image.open(path).convert("L").resize((400, 300), Image.BILINEAR)
    return {
        "sharpness": sharpness(gray),
        "luma": float(np.asarray(gray, dtype=np.float32).mean()),
        "hash": dhash(gray),
    }


def cull(
    root: Path, blur_ratio: float, min_luma: float, dup_bits: int, excluded: set[str]
) -> list[dict]:
    entries = []
    for session in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_")):
        photos = sorted(session.glob("img_*.jpg"))
        stats = [measure(p) for p in photos]
        if not stats:
            continue
        median_sharp = float(np.median([s["sharpness"] for s in stats]))
        last_kept = None
        for p, s in zip(photos, stats):
            reason = None
            if str(p.relative_to(root)) in excluded:
                reason = "excluded"
            elif s["luma"] < min_luma:
                reason = "dark"
            elif s["sharpness"] < blur_ratio * median_sharp:
                reason = "blurry"
            elif last_kept is not None and int((s["hash"] != last_kept).sum()) <= dup_bits:
                reason = "duplicate"
            if reason is None:
                last_kept = s["hash"]
            entries.append({
                "file": str(p.relative_to(root)),
                "session": session.name,
                "sharpness": round(s["sharpness"], 1),
                "sharpness_rel": round(s["sharpness"] / max(median_sharp, 1e-6), 2),
                "luma": round(s["luma"], 1),
                "keep": reason is None,
                "reason": reason,
            })
    return entries


def coco_images(root: Path, entries: list[dict]) -> dict:
    """Kept photos as a COCO file with no annotations, for teacher labeling."""
    images = []
    for i, e in enumerate((e for e in entries if e["keep"]), start=1):
        with Image.open(root / e["file"]) as im:
            w, h = im.size
        images.append({"id": i, "file_name": e["file"], "width": w, "height": h})
    return {"images": images, "annotations": [], "categories": []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="folder holding session_* directories")
    parser.add_argument("--out", type=Path, required=True, help="manifest JSON to write")
    parser.add_argument("--blur-ratio", type=float, default=0.35, help="drop below this x session median sharpness")
    parser.add_argument("--min-luma", type=float, default=20.0)
    parser.add_argument("--dup-bits", type=int, default=24, help="max differing hash bits (of 256) for a duplicate")
    parser.add_argument("--coco", type=Path,
                        help="also write the kept photos as a COCO image list (no annotations)")
    parser.add_argument("--exclude", type=Path,
                        help="text file of photos to drop by hand: '<session>/<file>  # why' per line")
    args = parser.parse_args()

    excluded = set()
    if args.exclude and args.exclude.exists():
        lines = (line.split("#")[0].strip() for line in args.exclude.read_text().splitlines())
        excluded = {line for line in lines if line}
    entries = cull(args.root, args.blur_ratio, args.min_luma, args.dup_bits, excluded)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"root": str(args.root), "blur_ratio": args.blur_ratio,
                                    "min_luma": args.min_luma, "dup_bits": args.dup_bits,
                                    "photos": entries}, indent=1))
    by_session: dict[str, list[dict]] = {}
    for e in entries:
        by_session.setdefault(e["session"], []).append(e)
    for name, es in by_session.items():
        reasons = {r: sum(e["reason"] == r for e in es) for r in ("blurry", "dark", "duplicate", "excluded")}
        print(f"{name}: {sum(e['keep'] for e in es)}/{len(es)} kept  " +
              "  ".join(f"{k}={v}" for k, v in reasons.items() if v))
    print(f"total kept: {sum(e['keep'] for e in entries)}/{len(entries)} -> {args.out}")
    if args.coco:
        args.coco.write_text(json.dumps(coco_images(args.root, entries)))
        print(f"COCO image list -> {args.coco}")


if __name__ == "__main__":
    main()
