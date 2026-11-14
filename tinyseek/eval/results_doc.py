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
SHORT = {"backpack": "bag"}  # dot-bar sublabels are narrow; the legend keeps the full name
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


def reviewed_teacher_stats(labels_dir: Path, verdicts_path: Path, thresholds: tuple[float, ...]) -> dict:
    """Teacher precision/recall per object against the hand-reviewed real labels.

    Ground truth = teacher boxes marked good + boxes drawn by hand. At threshold t a teacher
    box counts if its score >= t; precision = good / counted, recall = good / ground truth.
    Boxes without a verdict count as wrong, as the reviewer asked.
    """
    from tinyseek.label.curate import box_id

    export = json.loads(verdicts_path.read_text())
    verdicts = {(v["file"], v["box_id"]): v["verdict"] for v in export["verdicts"]}
    stats = {}
    for obj in OBJECTS:
        coco = json.loads((labels_dir / f"esp32cam_{obj}_raw.json").read_text())
        names = {img["id"]: img["file_name"] for img in coco["images"]}
        boxes = [(a["score"], verdicts.get((names[a["image_id"]], box_id(obj, a["bbox"]))) == "good")
                 for a in coco["annotations"]]
        drawn = sum(a["label"] == obj for a in export.get("added", []))
        truth = sum(good for _, good in boxes) + drawn
        per_t = {}
        for t in thresholds:
            counted = [good for score, good in boxes if score >= t]
            per_t[t] = {"tp": sum(counted), "counted": len(counted)}
        stats[obj] = {"truth": truth, "drawn": drawn, "per_t": per_t}
    return stats


def synth_yield(labels_dir: Path, synth_set: str, objects: tuple[str, ...], threshold: float) -> dict:
    """Share of generated images with at least one teacher box >= threshold, per object set."""
    out = {}
    for obj in objects:
        path = labels_dir / f"synth_{synth_set}_{obj}_raw.json"
        if not path.exists():
            continue
        coco = json.loads(path.read_text())
        hit = {a["image_id"] for a in coco["annotations"] if a["score"] >= threshold}
        out[obj] = {"k": len(hit), "n": len(coco["images"])}
    return out


SWEEP_OBJECTS = ("backpack", "cup")
SOURCE_VARIANTS = (  # image-source sweep run names -> chart labels
    ("real_only", "COCO real photos (reference)"),
    ("synth_only_v2first1k", "domain LoRA v2"),
    ("synth_only_plain1k", "plain SD"),
    ("synth_only_plain1k_camsim", "plain SD + camera sim"),
    ("synth_only_plain1k_sty075", "LoRA img2img 0.75"),
    ("synth_only_plain1k_sty075_camsim", "LoRA img2img 0.75 + camera sim"),
)
# Held-out camera sessions 013/014 (backpack; no generator or LoRA here saw them):
# (bar id, title, ((run name in the held-out summary, chart label), ...)).
NEAR, FAR = "heldout_qwen21_turbo_v2", "heldout_qwen21_turbo_far_v1"
HELDOUT_FIGURES = (
    ("heldout_backpack_training_set", "Backpack AP@0.5 on held-out camera photos, by training set (128 px)", (
        ("heldout_real_only", "COCO only (6,910 real)"),
        (NEAR, "Qwen near objects"),
        (f"{NEAR}_only_tone", "Qwen near, tone-matched"),
        (f"{NEAR}_only_tone_plus_real", "Qwen near tone + COCO"),
        (FAR, "Qwen far objects"),
        (f"{FAR}_only_tone", "Qwen far, tone-matched"),
        (f"{FAR}_only_tone_plus_real", "Qwen far tone + COCO"),
        (f"{NEAR}_only_tone_and_far", "Qwen near + far tone"),
        (f"{NEAR}_only_tone_and_far_plus_real", "Qwen near + far tone + COCO"))),
    ("heldout_camsim_ablation", "Camera-simulation parts on the same 1,000 Qwen images (held-out photos)", (
        (NEAR, "none (as generated)"),
        (f"{NEAR}_only_tone", "tone only"),
        (f"{NEAR}_only_vignette", "vignette only"),
        (f"{NEAR}_only_noise", "noise only"),
        (f"{NEAR}_only_distort", "barrel distortion only"),
        (f"{NEAR}_tone_vignette", "tone + vignette"),
        (f"{NEAR}_no_distort", "tone + vignette + noise"),
        (f"{NEAR}_camsim", "all four"))),
    ("heldout_data_efficiency", "Backpack AP@0.5 by number of synthetic images (Qwen far, tone-matched)", (
        (f"{FAR}_only_tone_n125", "125 images"),
        (f"{FAR}_only_tone_n250", "250 images"),
        (f"{FAR}_only_tone_n500", "500 images"),
        (f"{FAR}_only_tone", "1,000 images"))),
    ("heldout_input_size", "Backpack AP@0.5 by student input size (Qwen far, tone-matched)", (
        (f"{FAR}_only_tone", "128 px (31 M MACs)"),
        (f"{FAR}_only_tone_px160", "160 px (49 M MACs)"),
        (f"{FAR}_only_tone_px192", "192 px (71 M MACs)"),
        (f"{FAR}_only_tone_plus_real_px192", "192 px + COCO"),
        (f"{FAR}_only_tone_px256", "256 px (125 M, over budget)"))),
)


def seed_range(cell: dict) -> list[float]:
    return [round(min(cell["values"]), 4), round(max(cell["values"]), 4)]


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
    sweep = json.loads(Path(args.summary).read_text()) if Path(args.summary).exists() else {}

    def found_total(col: str) -> tuple[int, int]:
        k = sum(round(found(col)[o] * n_per_obj) for o in OBJECTS)
        return k, n_per_obj * len(OBJECTS)

    sims = {c: np.array([v for o in OBJECTS for v in clip[col_key(c)][o]]) for c in columns}
    k_ch, n_ch = found_total(chosen)
    rec_k = sum(round(r["value"] * r["n"]) for r in recall)
    rec_n = sum(r["n"] for r in recall)
    thresholds = (0.1, 0.2, 0.3)
    reviewed = reviewed_teacher_stats(Path(args.labels_dir), Path(args.verdicts), thresholds)
    rt = args.review_threshold
    rv_tp = sum(reviewed[o]["per_t"][rt]["tp"] for o in OBJECTS)
    rv_counted = sum(reviewed[o]["per_t"][rt]["counted"] for o in OBJECTS)
    rv_truth = sum(reviewed[o]["truth"] for o in OBJECTS)
    yields = synth_yield(Path(args.labels_dir), args.synth_set, ("cup", "backpack"), teacher["threshold"])
    y_k, y_n = sum(y["k"] for y in yields.values()), sum(y["n"] for y in yields.values())

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
        {"id": "teacher_precision_real_reviewed", "group": "Teacher on real ESP32 photos",
         "result": "teacher precision (hand-reviewed)",
         "metric": f"good / all teacher boxes at score >= {rt}, person/cup/backpack pooled", "n": rv_counted,
         "value": round(rv_tp / rv_counted, 4), "ci": wilson(rv_tp, rv_counted), "chance": None, "comparator": None,
         "compliance": "every teacher box on 508 kept photos reviewed by the owner in the gallery"},
        {"id": "teacher_recall_real_reviewed", "group": "Teacher on real ESP32 photos",
         "result": "teacher recall (hand-reviewed)",
         "metric": f"good teacher boxes at score >= {rt} / all true objects (incl. hand-drawn misses)",
         "n": rv_truth, "value": round(rv_tp / rv_truth, 4), "ci": wilson(rv_tp, rv_truth), "chance": None,
         "comparator": None, "compliance": "ground truth: owner-confirmed teacher boxes plus owner-drawn boxes, 590 photos"},
        {"id": "synth_label_yield", "group": "Synthetic sets", "result": "images the teacher labels",
         "metric": f"share of generated images with a box >= {teacher['threshold']}, cup and backpack pooled",
         "n": y_n, "value": round(y_k / max(1, y_n), 4), "ci": wilson(y_k, y_n), "chance": None, "comparator": None,
         "compliance": f"{args.synth_set}, 2,000 images per object"},
        *[{"id": f"student_esp32_{o}_synth_real_ap50", "group": "Student detector on real ESP32 photos",
           "result": f"AP@0.5 ({o}), synthetic + COCO training",
           "metric": "mean over 3 seeds, 590 hand-reviewed camera photos", "n": 590,
           "value": sweep[o]["esp32_real"]["synth_real"]["ap50"]["mean"],
           "ci": seed_range(sweep[o]["esp32_real"]["synth_real"]["ap50"]), "ci_label": "seed min-max",
           "chance": None,
           "comparator": {"label": "COCO only", "value": sweep[o]["esp32_real"]["real_only"]["ap50"]["mean"]},
           "compliance": "configs/synth_real.yaml vs real_only.yaml, seeds 0-2; interval = seed min-max; "
                         "the LoRA saw these photos (see notes)"}
          for o in SWEEP_OBJECTS if o in sweep],
        {"id": "student_cup_real_only_ap50", "group": "Student detector", "result": "COCO mAP@0.5 (cup)",
         "metric": "AP@0.5, 5,000 COCO val2017 images", "n": student["num_images"], "value": round(student["ap50"], 4),
         "ci": None, "chance": None,
         "comparator": {"label": "large cups only", "value": round(student["ap50_large"], 4)},
         "compliance": "configs/real_only.yaml prompt=cup seed=0, 128 px input, final epoch"},
    ]
    held = {}
    if Path(args.heldout_summary).exists():
        held = json.loads(Path(args.heldout_summary).read_text()).get("backpack", {}).get("esp32_heldout", {})
    for row_id, run, result, cmp_label, cmp_run, compliance in (
        ("student_heldout_backpack_synth_ap50", f"{FAR}_only_tone", "AP@0.5 (backpack), synthetic only, 128 px",
         "COCO only", "heldout_real_only",
         ("1,000 Qwen Image 2.1 Turbo images (far objects, tone-matched), configs/heldout_013_014.yaml, "
          "seeds 0-2; no real photos in generation or training")),
        ("student_heldout_backpack_px192_ap50", f"{FAR}_only_tone_px192", "AP@0.5 (backpack), synthetic only, 192 px",
         "128 px input", f"{FAR}_only_tone",
         ("same images and config with model.input_size=192 (71 M MACs, fits the budget check; "
          "on-device latency not measured)")),
    ):
        if run in held and cmp_run in held:
            headline.append({
                "id": row_id, "group": "Student detector on held-out ESP32 photos", "result": result,
                "metric": "mean over 3 seeds, 227 hand-reviewed photos of sessions 013/014", "n": 227,
                "value": held[run]["ap50"]["mean"], "ci": seed_range(held[run]["ap50"]), "ci_label": "seed min-max",
                "chance": None, "comparator": {"label": cmp_label, "value": held[cmp_run]["ap50"]["mean"]},
                "compliance": compliance})
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
    conditions = ("real_only", "teacher_real", "synth_only", "synth_real")
    for test_set, title in (("esp32_real", "Student AP@0.5 on real ESP32 photos (hand-reviewed)"),
                            ("coco_val", "Student AP@0.5 on COCO val2017")):
        rows = [{"label": f"{o} · {c}", "value": sweep[o][test_set][c]["ap50"]["mean"],
                 "ci": seed_range(sweep[o][test_set][c]["ap50"])}
                for o in SWEEP_OBJECTS if o in sweep for c in conditions if c in sweep[o].get(test_set, {})]
        top = max([r["ci"][1] for r in rows] + [0.01])
        bars.append({"id": f"student_ap50_{test_set}", "title": title, "unit": "AP@0.5", "ci_label": "seed min-max",
                     "domain": [0, round(min(1.0, top * 1.15), 2)], "better": "higher", "decimals": 3, "rows": rows})
    bars += [
        {"id": "teacher_precision_real_reviewed", "title": "Teacher precision on real ESP32 photos (hand-reviewed)",
         "unit": "precision", "domain": [0, 1], "better": "higher", "decimals": 2,
         "rows": [{"label": f"{o} · score >= {t}", "value": round(reviewed[o]["per_t"][t]["tp"]
                                                                  / max(1, reviewed[o]["per_t"][t]["counted"]), 4),
                   "ci": wilson(reviewed[o]["per_t"][t]["tp"], reviewed[o]["per_t"][t]["counted"])}
                  for o in OBJECTS for t in thresholds]},
        {"id": "teacher_recall_real_reviewed", "title": "Teacher recall on real ESP32 photos (hand-reviewed)",
         "unit": "recall", "domain": [0, 1], "better": "higher", "decimals": 2,
         "rows": [{"label": f"{o} · score >= {t}", "value": round(reviewed[o]["per_t"][t]["tp"]
                                                                  / max(1, reviewed[o]["truth"]), 4),
                   "ci": wilson(reviewed[o]["per_t"][t]["tp"], reviewed[o]["truth"])}
                  for o in OBJECTS for t in thresholds]},
        {"id": "synth_label_yield", "title": f"Synthetic images with a teacher box >= {teacher['threshold']}",
         "unit": "rate", "domain": [0, 1], "better": "higher", "decimals": 2,
         "rows": [{"label": f"{o} ({y['n']:,} images)", "value": round(y["k"] / y["n"], 4), "ci": wilson(y["k"], y["n"])}
                  for o, y in yields.items()]},
    ]
    source = {}
    if Path(args.source_summary).exists():
        source = json.loads(Path(args.source_summary).read_text()).get("backpack", {}).get("esp32_real", {})
    source_rows = [(c, label) for c, label in SOURCE_VARIANTS if c in source]
    if source_rows:
        bars.append({
            "id": "synth_source_backpack_esp32", "unit": "AP@0.5", "better": "higher", "decimals": 4,
            "title": "Synthetic-only backpack detector on real ESP32 photos, by image source (1,000 images)",
            "ci_label": "seed min-max",
            "domain": [0, round(max(max(source[c]["ap50"]["values"]) for c, _ in source_rows) * 1.15 + 0.005, 2)],
            "rows": [{"label": label, "value": source[c]["ap50"]["mean"], "ci": seed_range(source[c]["ap50"])}
                     for c, label in source_rows]})
    for bar_id, title, spec in HELDOUT_FIGURES:
        rows = [{"label": label, "value": held[c]["ap50"]["mean"], "ci": seed_range(held[c]["ap50"])}
                for c, label in spec if c in held]
        if rows:
            top = max(r["ci"][1] for r in rows)
            bars.append({"id": bar_id, "title": title, "unit": "AP@0.5", "better": "higher", "decimals": 3,
                         "ci_label": "seed min-max", "domain": [0, round(min(1.0, top * 1.15), 2)], "rows": rows})

    def ap(run: str) -> str:
        return f"{held[run]['ap50']['mean']:.3f}" if run in held else "n/a"

    dotbars = [{
        "id": "clip_sim_by_setting", "title": "CLIP similarity of generated images to real ESP32 photos",
        "unit": "cosine", "baseline": 0.5, "domain": [0.4, 1], "decimals": 3, "points_label": "generated images",
        "series_colors": {"base": "#888888", "lora": "#1f5fbf"},
        "variants": [{"key": o, "label": o, "fill": "solid"} for o in OBJECTS],
        "groups": [{"label": column_label(c), "color_key": "base" if c == "base" else "lora",
                    "bars": [{"variant": o, "sublabel": SHORT.get(o, o), "value": round(float(np.mean(clip[col_key(c)][o])), 4),
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
         "status_text": args.synth_set,
         "key_result": ", ".join(f"{o} {y['k'] / y['n']:.0%} labeled" for o, y in yields.items())
                       + f" at {teacher['threshold']}; in the gallery for review"},
        {"title": "Real ESP32 ground truth (hand-reviewed)", "status": "closed", "status_text": "590 photos",
         "key_result": ", ".join(f"{o} {reviewed[o]['truth']}" for o in OBJECTS)
                       + f" boxes; {sum(reviewed[o]['drawn'] for o in OBJECTS)} drawn by hand"},
        {"title": "Training conditions: cup and backpack, 3 seeds", "status": "closed", "status_text": "23 runs",
         "key_result": "; ".join(f"{o} on camera photos: COCO only {sweep[o]['esp32_real']['real_only']['ap50']['mean']:.3f}, "
                                 f"synthetic {sweep[o]['esp32_real']['synth_only']['ap50']['mean']:.3f}, "
                                 f"both {sweep[o]['esp32_real']['synth_real']['ap50']['mean']:.3f}"
                                 for o in SWEEP_OBJECTS if o in sweep)},
        {"title": "Synthetic image source (backpack, 1,000 images, 3 seeds)", "status": "closed",
         "status_text": "15 runs, 2026-11-10",
         "key_result": "domain-LoRA images work on camera photos; plain SD, camera simulation and LoRA "
                       "img2img stylizing all ~0: the gain is learned content, not the camera look"},
        {"title": "Domain LoRA v3 (851 photos incl. a second building)", "status": "closed",
         "status_text": "not adopted", "key_result": "barely closer to the camera, draws fewer objects (empty garages)"},
        {"title": "Held-out camera test (photos the LoRA never saw)", "status": "closed",
         "status_text": "2026-11-11, second machine",
         "key_result": "backpack sessions 013/014: LoRA that saw them 0.236, LoRA trained without them 0.043, "
                       "plain SD 0.006; most of the LoRA gain was leaked content"},
        {"title": "Larger generator: Qwen Image 2.1 Turbo, no LoRA (backpack, 1,000 images)", "status": "closed",
         "status_text": "2026-11-13",
         "key_result": f"held-out AP@0.5 {ap(NEAR)} from scene-description prompts alone; "
                       f"COCO only {ap('heldout_real_only')}"},
        {"title": "Camera-simulation ablation (same 1,000 Qwen images)", "status": "closed",
         "status_text": "24 runs, 2026-11-13",
         "key_result": f"tone matching helps ({ap(NEAR + '_only_tone')}); barrel distortion hurts "
                       f"({ap(NEAR + '_only_distort')}); vignette and noise within seed spread"},
        {"title": "Far-object prompts (objects small, as on the 160-degree lens)", "status": "closed",
         "status_text": "1,000 images, 2026-11-14",
         "key_result": f"held-out AP@0.5 {ap(FAR)} as generated, {ap(FAR + '_only_tone')} tone-matched, "
                       f"{ap(FAR + '_only_tone_plus_real')} with COCO added"},
        {"title": "Data efficiency and input size (far set, tone-matched)", "status": "closed",
         "status_text": "21 runs, 2026-11-14",
         "key_result": f"125 / 250 / 500 / 1,000 images: {ap(FAR + '_only_tone_n125')} / {ap(FAR + '_only_tone_n250')} / "
                       f"{ap(FAR + '_only_tone_n500')} / {ap(FAR + '_only_tone')}; input 160 / 192 / 256 px: "
                       f"{ap(FAR + '_only_tone_px160')} / {ap(FAR + '_only_tone_px192')} / {ap(FAR + '_only_tone_px256')}"},
        {"title": "On-device latency at 128 and 192 px", "status": "registered", "status_text": "next",
         "key_result": "decides whether the 192 px input can be adopted"},
        {"title": "Domain LoRA v3 (more photos of people)", "status": "registered",
         "status_text": "waiting for photos", "key_result": "v2 rarely draws people; person still uses v1"},
    ]
    notes = [
        ("Caveat: the domain LoRA that styled the synthetic images was trained on the same ESP32 photos used "
         "as the camera test set (including the backpack and cup sessions). The detector never saw them, but "
         "the image generator did, so the LoRA rows of the training-condition and image-source figures are "
         "overstated. The held-out figures below are free of this: they are scored on sessions 013/014, and "
         "the Qwen images come from text prompts only (no LoRA, no camera photos)."),
        ("Held-out figures: backpack, 227 hand-reviewed photos of sessions 013/014, synthetic sets of 1,000 "
         "images with raw teacher boxes at 0.3, bars = mean over seeds 0-2, whiskers = seed min-max. For "
         "comparison, SD 1.5 sets scored on the same photos on a second machine (2026-11-11, not in these "
         "figures): LoRA that saw the photos 0.236, LoRA trained without them 0.043, plain SD 0.006."),
        ("Held-out caveats: Qwen was given long scene descriptions and SD 1.5 short templates, so generator "
         "and prompt detail are not separated; the far-object prompts were sized from 24 backpack boxes in "
         "other sessions, not from the held-out ones; 160 and 192 px pass the model budget check but "
         "on-device latency is not measured, and 256 px does not fit; all of these detectors score about "
         "0.00-0.01 on COCO val2017. Qwen Image 2.1 is under a research, non-commercial licence."),
        ("Image-source sweep: synthetic-only detectors score ~0 on camera photos without the domain LoRA, "
         "even with the camera's distortion, colour and noise simulated; with it, 0.128 (1,000 images, this "
         "figure) to 0.200 (2,000 curated images: the backpack synth_only row of the training-condition "
         "figure). The LoRA's learned scenes and objects, from the test photos themselves, carry the gain."),
        "Student condition figures: bars = mean over seeds 0-2, whiskers = seed min-max; 128 px input.",
        "Generation: SD 1.5 (stable-diffusion-v1-5) + LCM-LoRA, 6 steps, fp32, 512x512, on a GTX 1080 Ti.",
        (f"LoRA comparison: {n_per_obj} images per object per setting, identical prompts and seeds; "
         f"teacher {teacher['name']} at {teacher['threshold']}."),
        ("'Teacher recall (one-object sessions)' is an upper bound: it assumes every photo in a session "
         "shows that object and does not check where the box is. The hand-reviewed figures replace it."),
        ("Hand review: the owner marked every teacher box on the real photos good or bad in the gallery and "
         "drew the objects the teacher missed; unmarked boxes count as wrong."),
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
    parser.add_argument("--summary", default="runs/summary.json", help="tinyseek.eval.summarize output")
    parser.add_argument("--heldout-summary", default="runs/summary_heldout_all.json",
                        help="summarize output of the held-out (sessions 013/014) runs")
    parser.add_argument("--source-summary", default="runs/summary_stylize_sweep.json",
                        help="summarize output of the synthetic image-source sweep")
    parser.add_argument("--verdicts", default="data/review/verdicts/real_esp32cam.json",
                        help="gallery verdict export for the real photos")
    parser.add_argument("--review-threshold", type=float, default=0.2)
    parser.add_argument("--synth-set", default="sd15_lcm_esp32v2")
    parser.add_argument("--synth-status", default="closed",
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
