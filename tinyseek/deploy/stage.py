"""Stage models, benchmark settings and parity cases for the ESP32-S3 evaluation firmware.

    python -m tinyseek.deploy.stage \\
        --model px192=runs/backpack/heldout_qwen21_turbo_far_v1_only_tone_px192/seed0 \\
        --model px128=runs/backpack/heldout_qwen21_turbo_far_v1_only_tone/seed0 \\
        --out runs/deploy/stage_backpack [--eval-set esp32_heldout] [--cases 8]

Writes ``<out>/sdcard/tinyseek_eval/`` (copy it to the card root) and the PC's expected
results under ``<out>/expected/`` (see ``tinyseek.deploy.layout``). Parity cases are
camera photos from an eval set of the first model's config, spread evenly over the
images that contain the prompt. Expected int8 outputs come from rebuilding esp-ppq's
simulated graph on the CPU (about 2 minutes per model; ``--no-sim`` skips it, and then
only preprocessing and decode can be compared). Reads the run directories, never writes them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tinyseek.config import fill, load_config, load_prompts
from tinyseek.data.dataset import Record, load_coco_records
from tinyseek.deploy.layout import CARD_ROOT, write_kv
from tinyseek.deploy.reference import io_bits, model_input
from tinyseek.deploy.simulate import sha256

NAME_RE = re.compile(r"^[a-z0-9_]{1,16}$")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    exponent: int


def graph_io(info_text: str) -> tuple[TensorInfo, TensorInfo]:
    """Input and output tensors (NHWC shape, dtype, exponent) from an esp-ppq ``model.info``."""
    m = re.search(r"graph \w+ \(\s*\n\s*%(\S+)\[(\w+), ([\dx]+)\], exponents: \[(-?\d+)\]", info_text)
    if not m:
        raise ValueError("no graph input found in model.info")
    inp = TensorInfo(m[1], tuple(int(v) for v in m[3].split("x")), m[2], int(m[4]))
    r = re.search(r"^\s*return %(\S+)\s*$", info_text, re.MULTILINE)
    if not r:
        raise ValueError("no graph output found in model.info")
    o = re.search(rf"^%{re.escape(r[1])}\[(\w+), ([\dx]+)\], exponents: \[(-?\d+)\]", info_text, re.MULTILINE)
    if not o:
        raise ValueError(f"no exponent listed for output {r[1]}")
    out = TensorInfo(r[1], tuple(int(v) for v in o[2].split("x")), o[1], int(o[3]))
    return inp, out


def parse_model_arg(text: str) -> tuple[str, Path]:
    name, sep, run = text.partition("=")
    if not sep or not NAME_RE.match(name):
        raise argparse.ArgumentTypeError(f"--model wants name=run_dir with name matching {NAME_RE.pattern}, got {text!r}")
    return name, Path(run)


def pick_cases(records: list[Record], n: int) -> list[Record]:
    """Evenly spaced over images with at least one box (deterministic, no seed needed)."""
    positives = [r for r in records if len(r.boxes)] or records
    if n >= len(positives):
        return positives
    idx = np.linspace(0, len(positives) - 1, n).round().astype(int)
    return [positives[i] for i in idx]


def eval_records(cfg: dict, eval_set: str) -> tuple[list[Record], int]:
    category_id = load_prompts()["prompts"][cfg["prompt"]]["coco_id"]
    spec = cfg["eval"]["sets"][eval_set]
    root = Path(cfg["data_root"])
    return load_coco_records(root / fill(spec["ann"], cfg), root / fill(spec["images"], cfg), category_id), category_id


def float_detections(run: Path, cfg: dict, cases: list[Record]) -> dict[int, list[dict]]:
    """The float checkpoint through tinyseek.eval.evaluate.predict, the PC's reported path."""
    from tinyseek.eval.evaluate import predict
    from tinyseek.train.model import build_model

    model = build_model(dict(cfg["model"], pretrained=False))
    model.load_state_dict(torch.load(run / "last.pt", map_location="cpu")["model"])
    ev = cfg["eval"]
    dets = predict(model, cases, cfg["model"]["input_size"], torch.device("cpu"), ev["score_threshold"],
                   ev["max_detections"], batch_size=8, num_workers=0)
    out: dict[int, list[dict]] = {r.image_id: [] for r in cases}
    for d in dets:
        out[d["image_id"]].append(d)
    return out


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stage_model(name: str, run: Path, card: Path, expected: Path, cases: list[tuple[str, Record]],
                scratch: Path | None) -> dict:
    cfg = load_config(run / "config.yaml")
    meta = json.loads((run / "model_meta.json").read_text())
    inp, out = graph_io((run / "model.info").read_text())
    size = meta["input_size"]
    if inp.shape != (1, size, size, 3):
        raise SystemExit(f"{run}: model.info input {inp.shape} does not match input_size {size}")
    in_bits, out_bits = io_bits(inp.dtype), io_bits(out.dtype)  # int16 for a w16a16 export
    shutil.copyfile(run / "model.espdl", card / "models" / f"{name}.espdl")
    ev = cfg["eval"]
    info = {
        "name": name,
        "run": str(run),
        "prompt": meta["prompt"],
        "input_size": size,
        "stride": meta["stride"],
        "num_classes": meta["num_classes"],
        "mean": ",".join(str(v) for v in meta["mean"]),
        "std": ",".join(str(v) for v in meta["std"]),
        "pad_value": 114,
        "score_threshold": ev["score_threshold"],
        "max_detections": ev["max_detections"],
        "input_exponent": inp.exponent,
        "output_exponent": out.exponent,
        "input_dtype": inp.dtype,
        "output_dtype": out.dtype,
        "quant": meta.get("quant", "w8a8"),
        "espdl_bytes": (run / "model.espdl").stat().st_size,
        "espdl_sha256": sha256(run / "model.espdl"),
    }
    write_kv(card / "models" / f"{name}.txt", info)

    exp_dir = expected / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    floats = float_detections(run, cfg, [r for _, r in cases])
    sim = None
    if scratch is not None:
        from tinyseek.deploy.simulate import rebuild

        sim = rebuild(run, scratch / name)
        if not sim.identical_to_deployed:
            raise SystemExit(f"{run}: rebuilt int8 graph differs from model.espdl; the simulation is no reference")
        if (sim.input_exponent, sim.output_exponent) != (inp.exponent, out.exponent):
            raise SystemExit(f"{run}: simulated exponents differ from model.info")
        if (sim.bits, sim.bits) != (in_bits, out_bits):
            raise SystemExit(f"{run}: simulated I/O width ({sim.bits} bits) differs from model.info")
    ppq_input_agrees = True
    for case_id, rec in cases:
        image = Image.open(rec.image_path).convert("RGB")
        q_in, lb = model_input(image, size, inp.exponent, in_bits)
        q_in.tofile(exp_dir / f"{case_id}_in.bin")
        (exp_dir / f"{case_id}_float.json").write_text(json.dumps(
            {"letterbox": [lb.scale, lb.pad_x, lb.pad_y], "detections": floats[rec.image_id]}, indent=1))
        if sim is not None:
            sim.run(q_in).tofile(exp_dir / f"{case_id}_out.bin")
            ppq_input_agrees &= _ppq_quantizes_like_us(sim, image, size, q_in)
    info.update({"int16_layers": sim.int16_ops if sim else None, "simulated": sim is not None,
                 "ppq_input_quantization_matches": ppq_input_agrees if sim else None})
    return info


def _ppq_quantizes_like_us(sim, image: Image.Image, size: int, q_in: np.ndarray) -> bool:
    """The executor fed the float tensor must give the same output as fed our int8 tensor."""
    from tinyseek.data.dataset import to_tensor
    from tinyseek.data.transforms import letterbox

    canvas, _ = letterbox(image, size)
    from_float = sim.executor(to_tensor(canvas)[None])[0][0].detach().numpy()
    from_ours = sim.executor(torch.from_numpy(q_in.astype(np.float32).transpose(2, 0, 1) * 2.0**sim.input_exponent)[None])[0][0]
    return bool(np.array_equal(from_float, from_ours.detach().numpy()))


def stage(models: list[tuple[str, Path]], out: Path, eval_set: str, n_cases: int, simulate: bool,
          bench: dict) -> dict:
    card = out / "sdcard" / CARD_ROOT
    for sub in ("models", "parity"):
        (card / sub).mkdir(parents=True, exist_ok=True)
    expected = out / "expected"
    cfg0 = load_config(models[0][1] / "config.yaml")
    records, _ = eval_records(cfg0, eval_set)
    cases = [(f"c{i:03d}", r) for i, r in enumerate(pick_cases(records, n_cases))]
    for case_id, rec in cases:
        shutil.copyfile(rec.image_path, card / "parity" / f"{case_id}.jpg")
        np.asarray(Image.open(rec.image_path).convert("RGB"), np.uint8).tofile(card / "parity" / f"{case_id}.rgb")
    (card / "parity" / "cases.txt").write_text("".join(f"{c}\n" for c, _ in cases))
    shutil.copyfile(cases[0][1].image_path, card / "bench.jpg")
    write_kv(card / "bench.txt", bench)

    scratch = out / "sim" if simulate else None
    infos = [stage_model(name, run, card, expected, cases, scratch) for name, run in models]
    manifest = {
        "git": git_rev(),
        "eval_set": eval_set,
        "bench": bench,
        "bench_jpg": str(cases[0][1].image_path),
        "cases": [{"id": c, "image": str(r.image_path), "image_id": r.image_id,
                   "size": list(Image.open(r.image_path).size), "gt_boxes": r.boxes.tolist()} for c, r in cases],
        "models": infos,
    }
    (out / "stage.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", type=parse_model_arg, required=True,
                        help="name=run_dir; the name becomes the file name on the card")
    parser.add_argument("--out", type=Path, default=Path("runs/deploy/stage"))
    parser.add_argument("--eval-set", default="esp32_heldout")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--no-sim", action="store_true", help="skip the esp-ppq rebuild (no expected outputs)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--camera", type=int, choices=[0, 1], default=0, help="1: also benchmark live capture")
    parser.add_argument("--camera-frames", type=int, default=50)
    parser.add_argument("--parity", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    names = [n for n, _ in args.model]
    if len(set(names)) != len(names):
        raise SystemExit("model names must be unique")
    bench = {"warmup": args.warmup, "runs": args.runs, "camera": args.camera,
             "camera_frames": args.camera_frames, "parity": args.parity}
    manifest = stage(args.model, args.out, args.eval_set, args.cases, not args.no_sim, bench)
    print(json.dumps({k: manifest[k] for k in ("eval_set", "bench")} |
                     {"models": [{k: m[k] for k in ("name", "input_size", "input_exponent", "output_exponent",
                                                    "simulated", "ppq_input_quantization_matches")}
                                 for m in manifest["models"]],
                      "cases": len(manifest["cases"]), "card": str(args.out / "sdcard" / CARD_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
