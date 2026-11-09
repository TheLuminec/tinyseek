"""Apply hand-review verdicts (from the dashboard gallery) to teacher labels.

    python -m tinyseek.label.curate --raw data/labels/owlv2/synth_sd15_lcm_esp32v2_cup_raw.json \
        --verdicts data/review/verdicts/synthetic_cup.json --threshold 0.3 \
        --out data/labels/owlv2/synth_sd15_lcm_esp32v2_cup_t0.3_curated.json

Rules per teacher box:
  verdict "good"  -> kept whatever its score (a person confirmed it), marked verified
  verdict "bad"   -> dropped whatever its score
  no verdict      -> kept only if score >= threshold, as with plain thresholding
                     (--unreviewed drop: never; for a set that was reviewed in full)
Boxes drawn by hand in the gallery (the export's "added" list) are added as verified
positives when their label is this file's category and their image is in it.
Teacher boxes are matched to verdicts by ``box_id``: a hash of the label and the box rounded to
whole pixels, so the id survives re-exporting or re-pushing the same teacher labels.

Verdicts file: {"verdicts": [{"file", "box_id", "verdict": "good" | "bad", ...}],
                "added": [{"file", "box_id", "label", "bbox": [x, y, w, h], ...}]}
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


def load_added(paths: list[Path]) -> list[dict]:
    """Boxes drawn by hand in the gallery (the export's "added" list)."""
    added: list[dict] = []
    for path in paths:
        added += json.loads(path.read_text()).get("added", [])
    return added


def curate(
    coco: dict, verdicts: dict[tuple[str, str], str], threshold: float, unreviewed: str = "threshold",
    added: list[dict] | None = None,
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
    # Hand-drawn boxes: owner-confirmed positives the teacher never produced.
    ids = {f: i for i, f in names.items()}
    cats = {name: cid for cid, name in labels.items()}
    counts.update(hand_added=0, hand_other_label=0, hand_unknown_image=0)
    for box in added or []:
        if box["label"] not in cats:
            counts["hand_other_label"] += 1
        elif box["file"] not in ids:
            counts["hand_unknown_image"] += 1
        else:
            x, y, w, h = box["bbox"]
            counts["hand_added"] += 1
            kept.append({"id": len(kept) + 1, "image_id": ids[box["file"]], "category_id": cats[box["label"]],
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0, "score": 1.0,
                         "verified": True, "source": "hand", "box_id": box["box_id"]})
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
                              args.unreviewed, load_added(args.verdicts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(curated))
    print(f"{args.out}: {len(curated['annotations'])} boxes  " + "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
