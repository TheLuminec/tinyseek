"""Write review indexes: one JSON per image set with each image's text, tags and teacher boxes.

    python -m tinyseek.eval.review_export real   --out data/review/esp32cam.json
    python -m tinyseek.eval.review_export synth  --prompt cup --synth-set sd15_lcm_esp32v2 \
        --out data/review/synthetic_cup_sd15_lcm_esp32v2.json

The dashboard's review page reads these, so it never has to parse the pipeline's own
formats. Boxes come from the raw teacher files (every box down to the teacher's score
floor), so a score slider on the page can reproduce any training threshold.

Index layout:
  {"project": "tinyseek", "set": ..., "kind": "real" | "synthetic", "image_root": <abs dir>,
   "labels": [...], "teacher": ..., "generated": <iso time>,
   "items": [{"file", "width", "height", "text", "seed", "session",
              "tags": {...}, "boxes": [{"id", "label", "score", "bbox": [x, y, w, h]}]}]}
Box ids come from tinyseek.label.curate.box_id, so gallery verdicts map back onto teacher boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from tinyseek.config import load_prompts
from tinyseek.label.curate import box_id


def all_labels() -> list[str]:
    """Every prompt in configs/prompts.yaml; label files that don't exist yet are skipped."""
    return list(load_prompts()["prompts"])



def load_boxes(label_files: dict[str, Path]) -> dict[str, list[dict]]:
    """file_name -> boxes, merged over the per-prompt raw teacher files that exist."""
    boxes: dict[str, list[dict]] = {}
    for label, path in label_files.items():
        if not path.exists():
            continue
        coco = json.loads(path.read_text())
        names = {img["id"]: img["file_name"] for img in coco["images"]}
        for a in sorted(coco["annotations"], key=lambda a: -a["score"]):
            bid = box_id(label, a["bbox"])
            image_boxes = boxes.setdefault(names[a["image_id"]], [])
            # Ids must be unique per image; two boxes with one id are the same box to the
            # pixel, so keep the higher-scoring one (curation gives both the same verdict).
            if any(b["id"] == bid for b in image_boxes):
                continue
            image_boxes.append({"id": bid, "label": label, "score": round(a["score"], 4),
                                "bbox": [round(v, 1) for v in a["bbox"]]})
    for bs in boxes.values():
        bs.sort(key=lambda b: -b["score"])
    return boxes


def size_of(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def real_index(data_root: Path, teacher: str) -> dict:
    cam = data_root / "esp32cam"
    manifest = json.loads((cam / "manifest.json").read_text())["photos"]
    captions = {}
    for name in ("captions_v2.jsonl", "captions.jsonl"):
        if (cam / name).exists():
            rows = [json.loads(line) for line in (cam / name).read_text().splitlines() if line.strip()]
            captions = {r["file"]: r["caption"] for r in rows}
            break
    sessions = {}
    if (cam / "sessions.csv").exists():
        with (cam / "sessions.csv").open() as f:
            sessions = {r["session"]: r["contents"] for r in csv.DictReader(f)}
    boxes = load_boxes({lb: data_root / "labels" / teacher / f"esp32cam_{lb}_raw.json" for lb in all_labels()})
    items = []
    for e in manifest:
        w, h = size_of(cam / "raw" / e["file"])
        items.append({
            "file": e["file"], "width": w, "height": h,
            "text": captions.get(e["file"]), "seed": None, "session": e["session"],
            "tags": {"kept": e["keep"], "reason": e["reason"], "luma": e["luma"],
                     "sharpness_rel": e["sharpness_rel"], "session_contents": sessions.get(e["session"])},
            "boxes": boxes.get(e["file"], []),
        })
    return {"set": "real/esp32cam", "kind": "real", "image_root": str((cam / "raw").resolve()), "items": items}


def synthetic_index(data_root: Path, prompt: str, synth_set: str, teacher: str) -> dict:
    set_dir = data_root / "synthetic" / prompt / synth_set
    meta = {}
    if (set_dir / "metadata.jsonl").exists():
        for line in (set_dir / "metadata.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                meta[r["file_name"]] = r
    boxes = load_boxes({prompt: data_root / "labels" / teacher / f"synth_{synth_set}_{prompt}_raw.json"})
    items = []
    for path in sorted(set_dir.glob("*.jpg")):
        m = meta.get(path.name, {})
        w, h = size_of(path)
        items.append({
            "file": path.name, "width": w, "height": h, "text": m.get("prompt"), "seed": m.get("seed"),
            "session": None, "tags": {"prompt": prompt}, "boxes": boxes.get(path.name, []),
        })
    return {"set": f"synthetic/{prompt}/{synth_set}", "kind": "synthetic",
            "image_root": str(set_dir.resolve()), "items": items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["real", "synth"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--teacher", default="owlv2")
    parser.add_argument("--prompt", help="synth: which prompt's set")
    parser.add_argument("--synth-set", help="synth: set name, e.g. sd15_lcm_esp32v2")
    args = parser.parse_args()

    if args.kind == "real":
        index = real_index(args.data_root, args.teacher)
    else:
        if not (args.prompt and args.synth_set):
            parser.error("synth needs --prompt and --synth-set")
        index = synthetic_index(args.data_root, args.prompt, args.synth_set, args.teacher)
    index = {"project": "tinyseek", **index, "labels": all_labels(), "teacher": args.teacher,
             "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index))
    n_boxes = sum(len(i["boxes"]) for i in index["items"])
    print(f"{args.out}: {len(index['items'])} images, {n_boxes} boxes ({index['set']})")


if __name__ == "__main__":
    main()
