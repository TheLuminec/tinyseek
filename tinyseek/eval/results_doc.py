"""Build the Avalon Monitor results document for tinyseek (`avm results push tinyseek <file>`).

    python -m tinyseek.eval.results_doc --out runs/results/tinyseek-results.json

Aggregates only, from runs already in this repo:
  - domain LoRA comparison (runs/lora/<run>/compare.json + the saved images): CLIP similarity
    to the real camera photos per image (bootstrap CI) and the teacher's found-rate per
    object (Wilson CI)
  - the teacher's found-rate on the real object sessions (data/labels/<teacher>/esp32cam_*)
  - the student baseline (runs/cup/real_only/seed0/eval.json)
Headline ids are stable: the dashboard keys per-metric history on them.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OBJECTS = ("person", "cup", "backpack")
# Real sessions that hold one object in every photo (data/esp32cam/sessions.csv).
OBJECT_SESSIONS = {"session_013": ("backpack", "black backpack"), "session_014": ("backpack", "gray backpack"),
                   "session_016": ("cup", "white frosted cup"), "session_017": ("cup", "clear glass cup")}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def bootstrap(values: np.ndarray, reps: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (reps, len(values))).mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def clip_per_image(run: Path, real_coco: Path, real_images: Path) -> dict[str, dict[str, list[float]]]:
    """Per-image CLIP similarity to the real-photo centroid, per column and object (cached, CPU)."""
    cache = run / "compare_clip.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import torch
    from PIL import Image

    from tinyseek.generate.compare_lora import ClipScorer

    clip = ClipScorer(torch.device("cpu"))
    files = [img["file_name"] for img in json.loads(real_coco.read_text())["images"]]
    centroid = torch.nn.functional.normalize(
        clip.images([Image.open(real_images / f).convert("RGB") for f in files]).mean(0), dim=0)
    out: dict[str, dict[str, list[float]]] = {}
    for col in sorted(p for p in (run / "compare").iterdir() if p.is_dir()):
        out[col.name] = {}
        for obj in OBJECTS:
            paths = sorted(col.glob(f"{obj}_*.png"))
            if paths:
                feats = clip.images([Image.open(p).convert("RGB") for p in paths])
                out[col.name][obj] = [round(float(v), 5) for v in feats @ centroid]
    cache.write_text(json.dumps(out))
    return out


def session_recall(labels_dir: Path, threshold: float) -> list[dict]:
    rows = []
    for session, (obj, name) in OBJECT_SESSIONS.items():
        coco = json.loads((labels_dir / f"esp32cam_{obj}_raw.json").read_text())
        ids = {img["id"] for img in coco["images"] if img["file_name"].startswith(session + "/")}
        best = dict.fromkeys(ids, 0.0)
        for a in coco["annotations"]:
            if a["image_id"] in best:
                best[a["image_id"]] = max(best[a["image_id"]], a["score"])
        k = sum(s >= threshold for s in best.values())
        rows.append({"label": name, "value": round(k / len(ids), 4), "ci": wilson(k, len(ids)), "n": len(ids)})
    return rows


def column_label(col: str) -> str:
    return {"base": "plain SD 1.5"}.get(col, col.replace("esp32cam_", "").replace("_", " "))


def build(args: argparse.Namespace) -> dict:
    run = Path(args.lora_run)
    compare = json.loads((run / "compare.json").read_text())
    teacher = compare["teacher"]
    clip = clip_per_image(run, Path(args.real), Path(args.real_images))
    n_per_obj = len(next(iter(next(iter(clip.values())).values())))

    def found(col: str) -> dict:
        return compare["columns"][col]["teacher_found"]

    def col_key(label: str) -> str:
        return label.replace(" ", "_")

    columns = list(compare["columns"])
    chosen, previous = args.chosen, args.previous
    recall = session_recall(Path(args.labels_dir), args.session_threshold)
    student = json.loads(Path(args.student_eval).read_text())["coco_val"]

    def found_total(col: str) -> tuple[int, int]:
        k = sum(round(found(col)[o] * n_per_obj) for o in OBJECTS)
        return k, n_per_obj * len(OBJECTS)

    sims = {c: np.array([v for o in OBJECTS for v in clip[col_key(c)][o]]) for c in columns}
    k_ch, n_ch = found_total(chosen)
    rec_k = sum(round(r["value"] * r["n"]) for r in recall)
    rec_n = sum(r["n"] for r in recall)

    headline = [
        {"id": "lora_clip_sim_real", "group": "Domain LoRA (camera look)", "result": "CLIP similarity to real photos",
         "metric": "mean cosine to the centroid of real ESP32 photos", "n": int(sims[chosen].size),
         "value": round(float(sims[chosen].mean()), 4), "ci": bootstrap(sims[chosen]), "chance": None,
         "comparator": {"label": column_label(previous), "value": round(float(sims[previous].mean()), 4)},
         "compliance": f"{column_label(chosen)}; SD 1.5 + LCM, same prompts and seeds per column"},
        {"id": "lora_teacher_found_rate", "group": "Domain LoRA (camera look)", "result": "teacher still finds the object",
         "metric": f"{teacher['name']} best box >= {teacher['threshold']}, person/cup/backpack pooled", "n": n_ch,
         "value": round(k_ch / n_ch, 4), "ci": wilson(k_ch, n_ch), "chance": None,
         "comparator": {"label": "plain SD 1.5", "value": round(found_total("base")[0] / found_total("base")[1], 4)},
         "compliance": column_label(chosen)},
        {"id": "teacher_recall_real_sessions", "group": "Teacher on real ESP32 photos", "result": "teacher recall",
         "metric": f"object found at score >= {args.session_threshold} (one-object sessions)", "n": rec_n,
         "value": round(rec_k / rec_n, 4), "ci": wilson(rec_k, rec_n), "chance": None, "comparator": None,
         "compliance": "upper bound: assumes the object is in every session photo; box not hand-checked"},
        {"id": "student_cup_real_only_ap50", "group": "Student detector", "result": "COCO mAP@0.5 (cup)",
         "metric": "AP@0.5, 5,000 COCO val2017 images", "n": student["num_images"], "value": round(student["ap50"], 4),
         "ci": None, "chance": None,
         "comparator": {"label": "large cups only", "value": round(student["ap50_large"], 4)},
         "compliance": "configs/real_only.yaml prompt=cup seed=0, 128 px input, final epoch"},
    ]
    bars = [
        {"id": "teacher_found_by_setting", "title": "Teacher found-rate in generated images",
         "unit": "rate", "domain": [0, 1], "better": "higher", "decimals": 2,
         "rows": [{"label": f"{o} · {column_label(c)}", "value": found(c)[o],
                   "ci": wilson(round(found(c)[o] * n_per_obj), n_per_obj)}
                  for o in OBJECTS for c in (["base", previous, chosen])]},
        {"id": "teacher_recall_sessions", "title": f"Teacher recall on real object sessions (score >= {args.session_threshold})",
         "unit": "rate", "domain": [0, 1], "better": "higher", "decimals": 2,
         "rows": [{"label": f"{r['label']} (n={r['n']})", "value": r["value"], "ci": r["ci"]} for r in recall]},
    ]
    dotbars = [{
        "id": "clip_sim_by_setting", "title": "CLIP similarity of generated images to real ESP32 photos",
        "unit": "cosine", "baseline": 0.5, "decimals": 3, "points_label": "generated images",
        "series_colors": {"base": "#888888", "lora": "#1f5fbf"},
        "variants": [{"key": o, "label": o, "fill": "solid"} for o in OBJECTS],
        "groups": [{"label": column_label(c), "color_key": "base" if c == "base" else "lora",
                    "bars": [{"variant": o, "sublabel": o, "value": round(float(np.mean(clip[col_key(c)][o])), 4),
                              "ci": bootstrap(np.array(clip[col_key(c)][o])), "points": clip[col_key(c)][o]}
                             for o in OBJECTS]} for c in columns],
    }]
    programmes = [
        {"title": "Student baseline (cup, real COCO only)", "status": "closed", "status_text": "seed 0, 2026-10-09",
         "key_result": f"AP@0.5 {student['ap50']:.3f} overall, {student['ap50_large']:.3f} on large cups"},
        {"title": "Domain LoRA v1 (130 photos)", "status": "closed", "status_text": "2026-11-06",
         "key_result": "camera look, but people and backpacks often missing"},
        {"title": "Domain LoRA v2 (508 photos)", "status": "closed", "status_text": "2026-11-08",
         "key_result": f"{column_label(chosen)}: CLIP sim {sims[chosen].mean():.3f}, found-rate {k_ch / n_ch:.2f}; "
                       "default for cup and backpack"},
        {"title": "Synthetic sets: cup and backpack (2,000 each)", "status": args.synth_status,
         "status_text": "sd15_lcm_esp32v2", "key_result": "for the synth_only and synth_real conditions"},
        {"title": "Domain LoRA v3 (more photos of people)", "status": "registered",
         "status_text": "waiting for photos", "key_result": "v2 rarely draws people; person still uses v1"},
    ]
    notes = [
        "Generation: SD 1.5 (stable-diffusion-v1-5) + LCM-LoRA, 6 steps, fp32, 512x512, on a GTX 1080 Ti.",
        (f"LoRA comparison: {n_per_obj} images per object per setting, identical prompts and seeds; "
         f"teacher {teacher['name']} at {teacher['threshold']}."),
        ("Teacher recall on real photos is an upper bound: it assumes every photo in a one-object session "
         "shows that object and does not check where the box is."),
    ]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()
    return {"schema": "tinyseek-results/1", "project": "tinyseek",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "commit": commit, "headline": headline, "programmes": programmes, "bars": bars,
            "distributions": [], "dotbars": dotbars, "notes": notes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lora-run", default="runs/lora/esp32cam_v2")
    parser.add_argument("--chosen", default="step 2000 x0.8", help="compare.json column in use")
    parser.add_argument("--previous", default="esp32cam_v1 step 500 x0.6", help="column to compare against")
    parser.add_argument("--real", default="data/esp32cam/kept_coco.json")
    parser.add_argument("--real-images", default="data/esp32cam/raw")
    parser.add_argument("--labels-dir", default="data/labels/owlv2")
    parser.add_argument("--session-threshold", type=float, default=0.2)
    parser.add_argument("--student-eval", default="runs/cup/real_only/seed0/eval.json")
    parser.add_argument("--synth-status", default="running",
                        choices=["running", "closed", "fail", "partial", "registered", "reference"])
    args = parser.parse_args()
    doc = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"{args.out}: {args.out.stat().st_size / 1024:.0f} KB, {len(doc['headline'])} headline rows")
    for h in doc["headline"]:
        print(f"  {h['id']}: {h['value']} ci={h['ci']} n={h['n']}")


if __name__ == "__main__":
    main()
