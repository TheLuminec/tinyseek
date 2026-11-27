"""Write a threshold-filtered copy of a raw teacher label file without re-running the teacher.

    python -m tinyseek.label.filter --raw data/labels/owlv2/synth_<set>_<prompt>_raw.json --threshold 0.15

The teacher already writes copies at its default thresholds; this adds another one. The threshold
cannot go below the raw file's own minimum score (``info.min_score``, 0.1 by default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyseek.label.teacher import filter_by_score, threshold_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="raw teacher output ending in _raw.json")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out", type=Path, help="default: the raw path with _t<threshold> in place of _raw")
    args = parser.parse_args()

    coco = json.loads(args.raw.read_text())
    floor = coco.get("info", {}).get("min_score")
    if floor is not None and args.threshold < floor:
        parser.error(f"threshold {args.threshold} is below the raw file's minimum score {floor}")
    filtered = filter_by_score(coco, args.threshold)
    out = args.out or threshold_path(args.raw, args.threshold)
    out.write_text(json.dumps(filtered))
    found = len({a["image_id"] for a in filtered["annotations"]})
    print(f"{out}: {len(filtered['annotations'])} boxes in {found} of {len(coco['images'])} images")


if __name__ == "__main__":
    main()
