"""Keep or drop camera sessions in a COCO label file.

    python -m tinyseek.data.split_sessions data/esp32cam/reviewed_labels.json \
        --exclude session_013 session_014 --out data/esp32cam/reviewed_labels_no013_014.json

Camera photos are named ``session_NNN/img_NNNNN.jpg``, so a session is the first path part.
Used to train on camera photos while scoring on sessions the training never saw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def session_of(file_name: str) -> str:
    return file_name.split("/", 1)[0]


def split(coco: dict, include: set[str] | None = None, exclude: set[str] | None = None) -> dict:
    """Images (and their annotations) whose session is in ``include`` and not in ``exclude``."""
    keep = {
        img["id"]
        for img in coco["images"]
        if (include is None or session_of(img["file_name"]) in include)
        and session_of(img["file_name"]) not in (exclude or set())
    }
    out = dict(coco, images=[i for i in coco["images"] if i["id"] in keep],
               annotations=[a for a in coco["annotations"] if a["image_id"] in keep])
    sessions = sorted({session_of(i["file_name"]) for i in out["images"]})
    out["info"] = dict(coco.get("info", {}), sessions=sessions)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("coco", type=Path)
    parser.add_argument("--include", nargs="*", help="sessions to keep (default: all)")
    parser.add_argument("--exclude", nargs="*", default=[], help="sessions to drop")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = split(json.loads(args.coco.read_text()), set(args.include) if args.include else None, set(args.exclude))
    args.out.write_text(json.dumps(out))
    print(f"{args.out}: {len(out['images'])} images, {len(out['annotations'])} boxes, sessions {out['info']['sessions']}")


if __name__ == "__main__":
    main()
