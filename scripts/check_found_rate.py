"""Exit 1 if the teacher found an object in fewer than MIN of a synthetic set's images.

    python scripts/check_found_rate.py data/labels/owlv2/synth_<set>_<prompt>_t0.3.json 0.8

A gate for generation chains: a pilot whose images the teacher cannot label is not worth 1,000 more.
"""
import json
import sys

coco = json.load(open(sys.argv[1]))
minimum = float(sys.argv[2])
found = len({a["image_id"] for a in coco["annotations"]})
rate = found / max(1, len(coco["images"]))
print(f"{sys.argv[1]}: teacher found the object in {found} of {len(coco['images'])} images ({rate:.0%})")
sys.exit(0 if rate >= minimum else 1)
