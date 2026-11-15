"""Compare what the board computed with what the PC expects, and summarize the benchmark.

    python -m tinyseek.deploy.compare --stage runs/deploy/stage_backpack \\
        --results /media/$USER/<card>/tinyseek_eval/results/run_001 [--json out.json]

Per model and case, four checks, each isolating one stage of the on-device pipeline:

- preprocess: the board's input tensor from ``<id>.rgb`` against the PC's, byte for byte
  (same frame pixels, so any difference is the letterbox/normalize/quantize code);
- model: the board's raw output for that input against esp-ppq's simulated int8 graph,
  in int8 steps (only meaningful when the inputs matched);
- decode: the board's detections against the PC decode of the board's *own* output
  tensor (isolates the post-processing code);
- end to end: detections from the board's own JPEG decode, and from the .rgb path,
  against the simulated graph's detections (IoU >= 0.5 matching), plus how far the
  board's JPEG decoder moves the input tensor.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tinyseek.data.transforms import Letterbox
from tinyseek.deploy.layout import read_kv
from tinyseek.deploy.reference import Detection, decode_output


def iou(a: Detection, b: Detection) -> float:
    ax0, ay0, ax1, ay1 = a.xyxy
    bx0, by0, bx1, by1 = b.xyxy
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def match(ref: list[Detection], got: list[Detection], iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one matching by descending reference score, as in AP matching."""
    used: set[int] = set()
    pairs = []
    for r in sorted(ref, key=lambda d: -d.score):
        best, best_iou = None, iou_threshold
        for j, g in enumerate(got):
            if j not in used and g.cls == r.cls and (v := iou(r, g)) >= best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
            pairs.append((r, got[best], best_iou))
    n_ref, n_got, n = len(ref), len(got), len(pairs)
    return {
        "ref": n_ref,
        "board": n_got,
        "matched": n,
        "f1": 1.0 if n_ref + n_got == 0 else 2 * n / (n_ref + n_got),
        "mean_iou": float(np.mean([p[2] for p in pairs])) if pairs else None,
        "max_score_diff": float(max(abs(p[0].score - p[1].score) for p in pairs)) if pairs else None,
    }


def decode_identical(ref: list[Detection], got: list[Detection], px: float = 1e-3, score: float = 1e-5) -> bool:
    """Same detections in the same order, up to float32 rounding differences."""
    if len(ref) != len(got):
        return False
    key = lambda d: (-round(d.score, 4), d.x, d.y)
    return all(
        a.cls == b.cls and abs(a.score - b.score) <= score
        and max(abs(a.x - b.x), abs(a.y - b.y), abs(a.w - b.w), abs(a.h - b.h)) <= px
        for a, b in zip(sorted(ref, key=key), sorted(got, key=key))
    )


def read_board_dets(path: Path) -> dict[tuple[str, str], list[Detection]]:
    out: dict[tuple[str, str], list[Detection]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            out[(row["case"], row["path"])].append(Detection(
                float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]), float(row["score"]), int(row["cls"])))
    return out


def tensor_diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return {"differing": int((d > 0).sum()), "of": int(d.size), "max_abs": int(d.max()), "mean_abs": float(d.mean()),
            "within_1": float((d <= 1).mean())}


def load(path: Path) -> np.ndarray | None:
    return np.fromfile(path, np.int8) if path.exists() else None


def compare_case(model: dict, case_id: str, expected: Path, board: Path, dets: dict) -> dict:
    size, grid = int(model["input_size"]), int(model["input_size"]) // int(model["stride"])
    nc, out_exp = int(model["num_classes"]), int(model["output_exponent"])
    ch = nc + 4
    fl = json.loads((expected / f"{case_id}_float.json").read_text())
    lb = Letterbox(*fl["letterbox"])
    kw = {"num_classes": nc, "stride": int(model["stride"]), "score_threshold": float(model["score_threshold"]),
              "max_detections": int(model["max_detections"])}
    res: dict = {"case": case_id}
    exp_in, exp_out = load(expected / f"{case_id}_in.bin"), load(expected / f"{case_id}_out.bin")
    sim_dets = decode_output(exp_out.reshape(grid, grid, ch), out_exp, lb, **kw) if exp_out is not None else None
    floats = [Detection(*d["bbox"], d["score"]) for d in fl["detections"]]
    if sim_dets is not None:
        res["sim_vs_float"] = match(floats, sim_dets)

    for path in ("rgb", "jpg"):
        b_in, b_out = load(board / f"{case_id}_{path}_in.bin"), load(board / f"{case_id}_{path}_out.bin")
        r: dict = {}
        if b_in is None or b_out is None:
            res[path] = {"missing": True}
            continue
        if b_in.size != size * size * 3 or b_out.size != grid * grid * ch:
            res[path] = {"bad_size": [int(b_in.size), int(b_out.size)]}
            continue
        r["input"] = tensor_diff(b_in, exp_in)
        if exp_out is not None:
            r["output"] = tensor_diff(b_out, exp_out)
        board_dets = dets.get((case_id, path), [])
        pc_decode = decode_output(b_out.reshape(grid, grid, ch), out_exp, lb, **kw)
        r["decode_identical"] = decode_identical(pc_decode, board_dets)
        if sim_dets is not None:
            r["dets_vs_sim"] = match(sim_dets, board_dets)
        r["dets_vs_float"] = match(floats, board_dets)
        res[path] = r
    return res


def summarize_model(cases: list[dict]) -> dict:
    def collect(path: str, key: str) -> list:
        return [c[path][key] for c in cases if key in c.get(path, {})]

    rgb_in = collect("rgb", "input")
    out = {
        "cases": len(cases),
        "preprocess_exact": f"{sum(d['differing'] == 0 for d in rgb_in)}/{len(rgb_in)}",
        "decode_identical": f"{sum(collect('rgb', 'decode_identical') + collect('jpg', 'decode_identical'))}"
                            f"/{len(collect('rgb', 'decode_identical') + collect('jpg', 'decode_identical'))}",
    }
    rgb_out = [c["rgb"]["output"] for c in cases
               if "output" in c.get("rgb", {}) and c["rgb"]["input"]["differing"] == 0]
    if rgb_out:
        out["model_output_exact"] = f"{sum(d['differing'] == 0 for d in rgb_out)}/{len(rgb_out)}"
        out["model_output_max_abs_lsb"] = max(d["max_abs"] for d in rgb_out)
        out["model_output_within_1_lsb"] = float(np.mean([d["within_1"] for d in rgb_out]))
    for path in ("rgb", "jpg"):
        m = collect(path, "dets_vs_sim")
        if m:
            out[f"{path}_dets_vs_sim_f1"] = float(np.mean([x["f1"] for x in m]))
    jpg_in = collect("jpg", "input")
    if jpg_in:
        out["jpeg_input_mean_abs_lsb"] = float(np.mean([d["mean_abs"] for d in jpg_in]))
        out["jpeg_input_max_abs_lsb"] = max(d["max_abs"] for d in jpg_in)
    return out


def bench_table(results: Path) -> list[dict]:
    path = results / "bench.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def compare(stage_dir: Path, results: Path) -> dict:
    manifest = json.loads((stage_dir / "stage.json").read_text())
    report: dict = {"stage": str(stage_dir), "results": str(results), "models": {}}
    for m in manifest["models"]:
        name = m["name"]
        board = results / "parity" / name
        dets = read_board_dets(board / "dets.csv")
        cases = [compare_case(m, c["id"], stage_dir / "expected" / name, board, dets) for c in manifest["cases"]]
        report["models"][name] = {"summary": summarize_model(cases), "cases": cases}
    report["bench"] = bench_table(results)
    meta = results / "run.txt"
    report["board"] = read_kv(meta) if meta.exists() else {}
    return report


def print_report(report: dict) -> None:
    if report["board"]:
        print("board: " + ", ".join(f"{k}={v}" for k, v in report["board"].items()))
    for name, m in report["models"].items():
        print(f"\n[{name}]")
        for k, v in m["summary"].items():
            print(f"  {k:28s} {v}")
    rows = report["bench"]
    if rows:
        print(f"\n{'model':10s} {'stage':14s} {'runs':>5s} {'mean_ms':>9s} {'min_ms':>9s} {'max_ms':>9s} {'std_ms':>8s} {'fps':>6s}")
        for r in rows:
            print(f"{r['model']:10s} {r['stage']:14s} {r['runs']:>5s} {r['mean_ms']:>9s} {r['min_ms']:>9s} "
                  f"{r['max_ms']:>9s} {r['std_ms']:>8s} {r.get('fps', ''):>6s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=Path, required=True, help="the --out directory of tinyseek.deploy.stage")
    parser.add_argument("--results", type=Path, required=True, help="a results/run_NNN folder copied off the card")
    parser.add_argument("--json", type=Path, help="write the full per-case report here")
    args = parser.parse_args()
    report = compare(args.stage, args.results)
    print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
