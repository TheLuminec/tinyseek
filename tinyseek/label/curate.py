"""Apply hand-review verdicts (from the dashboard gallery) to teacher labels.

    python -m tinyseek.label.curate --raw data/labels/owlv2/synth_sd15_lcm_esp32v2_cup_raw.json \
        --verdicts data/review/verdicts/synthetic_cup.json --threshold 0.3 \
        --out data/labels/owlv2/synth_sd15_lcm_esp32v2_cup_t0.3_curated.json

Rules per teacher box:
  verdict "good"  -> kept whatever its score (a person confirmed it), marked verified
  verdict "bad"   -> dropped whatever its score
  no verdict      -> kept only if score >= threshold, as with plain thresholding
                     (--unreviewed drop: never; for a set that was reviewed in full)
Boxes are matched to verdicts by ``box_id``: a hash of the label and the box rounded to
whole pixels, so the id survives re-exporting or re-pushing the same teacher labels.

Verdicts file: {"verdicts": [{"file": ..., "box_id": ..., "verdict": "good" | "bad", ...}]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def box_id(label: str, bbox: list[float]) -> str:
    """Stable 8-hex id for a teacher box: label + bbox [x, y, w, h] rounded to whole pixels."""
    key = f"{label}|" + ",".join(str(round(v)) for v in bbox)
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def load_verdicts(paths: list[Path]) -> dict[tuple[str, str], str]:
    verdicts: dict[tuple[str, str], str] = {}
    for path in paths:
        for v in json.loads(path.read_text())["verdicts"]:
            verdicts[(v["file"], v["box_id"])] = v["verdict"]
    return verdicts


def curate(
    coco: dict, verdicts: dict[tuple[str, str], str], threshold: float, unreviewed: str = "threshold"
) -> tuple[dict, dict[str, int]]:
    names = {img["id"]: img["file_name"] for img in coco["images"]}
    labels = {c["id"]: c["name"] for c in coco["categories"]}
    kept, counts = [], {"good": 0, "bad": 0, "threshold_kept": 0, "threshold_dropped": 0}
    for a in coco["annotations"]:
        verdict = verdicts.get((names[a["image_id"]], box_id(labels[a["category_id"]], a["bbox"])))
        if verdict == "bad":
            counts["bad"] += 1
            continue
        if verdict == "good":
            counts["good"] += 1
            kept.append(dict(a, id=len(kept) + 1, verified=True))
        elif unreviewed == "threshold" and a["score"] >= threshold:
            counts["threshold_kept"] += 1
            kept.append(dict(a, id=len(kept) + 1, verified=False))
        else:
            counts["threshold_dropped"] += 1
    info = dict(coco.get("info", {}), curated={"threshold": threshold, "unreviewed": unreviewed, **counts})
    return dict(coco, info=info, annotations=kept), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="raw teacher label file (COCO with scores)")
    parser.add_argument("--verdicts", type=Path, nargs="*", default=[], help="verdict files from the gallery")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--unreviewed", choices=["threshold", "drop"], default="threshold",
                        help="drop: keep only boxes marked good (a fully reviewed set)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    curated, counts = curate(json.loads(args.raw.read_text()), load_verdicts(args.verdicts), args.threshold,
                              args.unreviewed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(curated))
    print(f"{args.out}: {len(curated['annotations'])} boxes  " + "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
