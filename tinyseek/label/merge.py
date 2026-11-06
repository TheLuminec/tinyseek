"""Merge per-prompt teacher label files into one multi-class COCO file for review.

    python -m tinyseek.label.merge --out data/esp32cam/draft_labels.json \
        data/labels/owlv2/esp32cam_person_raw.json:0.15 \
        data/labels/owlv2/esp32cam_cup_raw.json:0.2 \
        data/labels/owlv2/esp32cam_backpack_raw.json:0.3

Each input is ``<raw label file>:<score threshold>``; all inputs must label the same
image list. The output keeps each box's teacher score and opens in COCO-capable
annotation tools (CVAT, Label Studio) for correction into a ground-truth test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(inputs: list[tuple[Path, float]]) -> dict:
    images, categories, annotations = None, [], []
    for path, threshold in inputs:
        coco = json.loads(path.read_text())
        ids = [img["id"] for img in coco["images"]]
        if images is None:
            images = coco["images"]
        elif ids != [img["id"] for img in images]:
            raise SystemExit(f"{path} labels a different image list")
        categories += [c for c in coco["categories"] if c not in categories]
        for a in coco["annotations"]:
            if a["score"] >= threshold:
                annotations.append(dict(a, id=len(annotations) + 1))
    return {"info": {"inputs": [f"{p}:{t}" for p, t in inputs], "draft": True},
            "images": images or [], "annotations": annotations, "categories": categories}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="<raw label file>:<threshold>")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pairs = []
    for item in args.inputs:
        path, _, thr = item.rpartition(":")
        pairs.append((Path(path), float(thr)))
    merged = merge(pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged))
    names = {c["id"]: c["name"] for c in merged["categories"]}
    counts = {n: sum(a["category_id"] == i for a in merged["annotations"]) for i, n in names.items()}
    print(f"{args.out}: {len(merged['images'])} images, " + ", ".join(f"{n} {k}" for n, k in counts.items()))


if __name__ == "__main__":
    main()
