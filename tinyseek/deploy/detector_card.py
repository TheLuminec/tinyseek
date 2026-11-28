"""Stage the SD card for the live detector (firmware/detector).

    python -m tinyseek.deploy.detector_card \\
        --model runs/backpack/heldout_qwen21_turbo_far_v1_only_tone_px192/seed0 \\
        [--model cup=runs/cup/.../seed0] [--threshold 0.4 | --threshold backpack=0.4] \\
        [--prompt backpack] [--source camera|replay] [--set decode_scale=1 --set resize=bilinear] \\
        [--replay-ann data/esp32cam/reviewed_labels_s013_014.json --replay-images data/esp32cam/raw \\
         [--replay-limit 50]] \\
        --out runs/deploy/detector_card

Writes ``<out>/sdcard/tinyseek/`` (copy the ``tinyseek`` folder to the card root) and
``<out>/card.json``, which records what was staged and, for a replay set, which label file
and image ids the replay files stand for (``tinyseek.deploy.replay`` scores against it).

Card layout::

    /tinyseek/config.txt             key=value settings (firmware/detector/README.md)
    /tinyseek/models/<prompt>.espdl  one ESP-DL model per prompt; the prompt is the file name
    /tinyseek/models/<prompt>.txt    its sidecar: input size (input_size for a square model,
                                     input_width and input_height for a rectangular one),
                                     stride, exponents, mean/std,
                                     score_threshold (decode, as in the PC evaluation),
                                     max_detections, detect_threshold (LED), provenance
    /tinyseek/replay/NNNNN.jpg       replay set (camera JPEGs, renamed in label-file order)
    /tinyseek/results/run_NNN/       written by the board in replay mode

Each ``--model`` is a run directory from tinyseek.train.export (model.espdl,
model_meta.json; model.info and config.yaml are used when present). The detect
threshold comes from ``--threshold``, else ``detect_threshold`` in model_meta.json, else
``eval.detect_threshold`` in the run's config.yaml, else 0.3. Wi-Fi credentials are never
written here: add them to config.txt on the card (see the README).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image

from tinyseek.config import fields_wh, size_fields
from tinyseek.deploy.layout import write_kv

CARD_DIR = "tinyseek"
PROMPT_RE = re.compile(r"^[a-z0-9_]{1,32}$")
DEFAULT_DETECT_THRESHOLD = 0.3
FRAME_W, FRAME_H = 800, 600  # the board's frame buffers are sized for the camera's SVGA

# config.txt keys the firmware reads (firmware/detector/src/detector_logic.cpp), with the
# staged defaults. Credentials are deliberately absent: they go on the card only.
CONFIG_DEFAULTS = {
    "prompt": "",
    "source": "camera",
    "decode_scale": "0",
    "resize": "pillow",
    "decoder": "tjpgd",
    "multicore": "0",
    "internal_kb": "0",
    "pipeline": "0",
    "led": "both",
    "print_min_score": "0.1",
    "wifi": "off",
    "http_port": "80",
    "camera_warmup_ms": "4000",
    "replay_dir": "/sdcard/tinyseek/replay",
    "replay_limit": "0",
    "hostname": "tinyseek",
    "ap_ssid": "tinyseek",
}
SECRET_KEYS = {"wifi_ssid", "wifi_password", "ap_password"}

CONFIG_FOOTER = """
# Wi-Fi is off unless set here, on the card only (never in the repository):
#   wifi=sta
#   wifi_ssid=<network name>
#   wifi_password=<password>
# or the board as its own access point:
#   wifi=ap
#   ap_ssid=tinyseek
#   ap_password=<8 to 63 characters>
"""


def parse_model_arg(text: str) -> tuple[str | None, Path]:
    name, sep, run = text.partition("=")
    if sep and PROMPT_RE.match(name):
        return name, Path(run)
    return None, Path(text)


def parse_thresholds(values: list[str]) -> tuple[float | None, dict[str, float]]:
    every, per = None, {}
    for v in values:
        name, sep, num = v.partition("=")
        try:
            if sep:
                per[name] = float(num)
            else:
                every = float(v)
        except ValueError:
            raise SystemExit(f"--threshold wants 0.4 or prompt=0.4, got {v!r}") from None
    for t in [every, *per.values()]:
        if t is not None and not 0.0 <= t <= 1.0:
            raise SystemExit(f"thresholds must be in [0, 1], got {t}")
    return every, per


def parse_sets(values: list[str]) -> dict[str, str]:
    out = {}
    for v in values:
        key, sep, val = v.partition("=")
        if not sep:
            raise SystemExit(f"--set wants key=value, got {v!r}")
        if key in SECRET_KEYS:
            raise SystemExit(f"{key} is a credential: write it into config.txt on the card, not here")
        if key not in CONFIG_DEFAULTS:
            raise SystemExit(f"unknown config key {key!r}; known: {', '.join(sorted(CONFIG_DEFAULTS))}")
        out[key] = val
    return out


def _run_config(run: Path) -> dict:
    if not (run / "config.yaml").exists():
        return {}
    from tinyseek.config import load_config

    return load_config(run / "config.yaml")


def model_sidecar(prompt: str, run: Path, threshold: float | None) -> dict:
    """The models/<prompt>.txt values for one exported run."""
    from tinyseek.deploy.simulate import sha256

    meta = json.loads((run / "model_meta.json").read_text())
    cfg = _run_config(run)
    ev = cfg.get("eval", {})
    if threshold is not None:
        detect, source = threshold, "cli"
    elif "detect_threshold" in meta:
        detect, source = float(meta["detect_threshold"]), "model_meta"
    elif "detect_threshold" in ev:
        detect, source = float(ev["detect_threshold"]), "config"
    else:
        detect, source = DEFAULT_DETECT_THRESHOLD, "default"
    wh = fields_wh(meta)  # (width, height), from input_size or input_width/input_height
    info = {
        "prompt": prompt,
        "run": str(run),
        **size_fields(wh),  # input_size for a square model, input_width + input_height otherwise
        "stride": meta["stride"],
        "num_classes": meta["num_classes"],
        "mean": ",".join(str(v) for v in meta["mean"]),
        "std": ",".join(str(v) for v in meta["std"]),
        "pad_value": 114,
        "score_threshold": ev.get("score_threshold", 0.05),
        "max_detections": ev.get("max_detections", 20),
        "detect_threshold": detect,
        "detect_threshold_source": source,
    }
    if (run / "model.info").exists():
        from tinyseek.deploy.reference import io_bits
        from tinyseek.deploy.stage import check_io_shapes, graph_io

        inp, out = graph_io((run / "model.info").read_text())
        check_io_shapes(run, inp, out, wh, meta["stride"])
        for dtype in (inp.dtype, out.dtype):
            io_bits(dtype)  # the board runs INT8 or INT16 I/O only
        info.update(input_exponent=inp.exponent, output_exponent=out.exponent, input_dtype=inp.dtype,
                    output_dtype=out.dtype)
    info["quant"] = meta.get("quant", "w8a8")
    info.update(espdl_bytes=(run / "model.espdl").stat().st_size, espdl_sha256=sha256(run / "model.espdl"))
    return info


def board_decodable(path: Path) -> str | None:
    """None if the board can decode this JPEG into its frame buffer, else why not."""
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return "not a JPEG"
            if im.info.get("progressive") or im.info.get("progression"):
                return "progressive JPEG"
            if im.mode != "RGB":
                return f"{im.mode} JPEG (TJpgDec needs 3 components)"
            if im.width > FRAME_W or im.height > FRAME_H:
                return f"{im.width}x{im.height} is larger than the {FRAME_W}x{FRAME_H} frame buffer"
    except OSError as e:
        return f"unreadable: {e}"
    from tinyseek.deploy import tjpgd

    if tjpgd.available():
        try:
            tjpgd.decode(path, 3)
        except tjpgd.TjpgdError as e:
            return str(e)
    return None


def stage_replay(ann: Path, images: Path, card: Path, limit: int, out: Path) -> dict:
    coco = json.loads(ann.read_text())
    replay = card / "replay"
    replay.mkdir(parents=True, exist_ok=True)
    files, skipped, kept_ids = [], [], []
    for img in coco["images"]:
        if limit and len(files) >= limit:
            break
        src = images / img["file_name"]
        why = board_decodable(src) if src.exists() else "missing"
        if why:
            skipped.append({"image": img["file_name"], "reason": why})
            continue
        name = f"{len(files):05d}.jpg"
        shutil.copyfile(src, replay / name)
        files.append({"file": name, "image_id": img["id"], "image": img["file_name"]})
        kept_ids.append(img["id"])
    keep = set(kept_ids)
    subset = dict(coco, images=[i for i in coco["images"] if i["id"] in keep],
                  annotations=[a for a in coco["annotations"] if a["image_id"] in keep])
    (out / "replay_labels.json").write_text(json.dumps(subset))
    return {"ann": str(ann), "images": str(images), "labels": "replay_labels.json", "files": files,
            "skipped": skipped}


def stage(models: list[tuple[str | None, Path]], out: Path, every: float | None, per: dict[str, float],
          prompt: str | None, sets: dict[str, str], replay_ann: Path | None, replay_images: Path | None,
          replay_limit: int, replay_category: int | None) -> dict:
    card = out / "sdcard" / CARD_DIR
    if card.exists():
        shutil.rmtree(card)  # a stale model or replay file must not survive a restage
    (card / "models").mkdir(parents=True)
    staged = []
    for name, run in models:
        meta = json.loads((run / "model_meta.json").read_text())
        name = name or meta["prompt"]
        if not PROMPT_RE.match(name):
            raise SystemExit(f"{run}: prompt {name!r} is not a valid model name ([a-z0-9_]{{1,32}}); use name=run_dir")
        if name in {m["prompt"] for m in staged}:
            raise SystemExit(f"two models for prompt {name!r}: give one a name with name=run_dir")
        info = model_sidecar(name, run, per.get(name, every))
        shutil.copyfile(run / "model.espdl", card / "models" / f"{name}.espdl")
        write_kv(card / "models" / f"{name}.txt", info)
        staged.append(info)
    unknown = set(per) - {m["prompt"] for m in staged}
    if unknown:
        raise SystemExit(f"--threshold for prompts that are not staged: {sorted(unknown)}")
    active = prompt or staged[0]["prompt"]
    if active not in {m["prompt"] for m in staged}:
        raise SystemExit(f"--prompt {active!r} is not one of the staged models")

    config = {**CONFIG_DEFAULTS, "prompt": active, **sets}
    lines = ["# tinyseek detector settings: key=value, one per line (firmware/detector/README.md)"]
    lines += [f"{k}={v}" for k, v in config.items()]
    (card / "config.txt").write_text("\n".join(lines) + "\n" + CONFIG_FOOTER)

    manifest: dict = {"card": str(card), "config": config, "models": staged}
    try:
        from tinyseek.deploy.stage import git_rev

        manifest["git"] = git_rev()
    except ImportError:
        pass
    if replay_ann is not None:
        if replay_images is None:
            raise SystemExit("--replay-ann needs --replay-images")
        rep = stage_replay(replay_ann, replay_images, card, replay_limit, out)
        if replay_category is None:
            from tinyseek.config import load_prompts

            prompts = load_prompts()["prompts"]
            if active not in prompts:
                raise SystemExit(f"prompt {active!r} is not in configs/prompts.yaml: give --replay-category")
            replay_category = prompts[active]["coco_id"]
        rep.update(prompt=active, category_id=replay_category)
        manifest["replay"] = rep
    (out / "card.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, type=parse_model_arg,
                        help="run_dir, or prompt=run_dir to name the model (default: model_meta.json prompt)")
    parser.add_argument("--threshold", action="append", default=[],
                        help="detect threshold for the LED and the web page: 0.4 (all) or prompt=0.4")
    parser.add_argument("--prompt", help="the active prompt in config.txt (default: the first model)")
    parser.add_argument("--source", choices=["camera", "replay"], help="config.txt source (default: camera)")
    parser.add_argument("--set", action="append", default=[], help="another config.txt key=value")
    parser.add_argument("--replay-ann", type=Path, help="COCO label file of the replay photos")
    parser.add_argument("--replay-images", type=Path, help="folder the label file's file_name entries are under")
    parser.add_argument("--replay-limit", type=int, default=0, help="stage at most this many replay images")
    parser.add_argument("--replay-category", type=int, help="COCO category id to score (default: from prompts.yaml)")
    parser.add_argument("--out", type=Path, default=Path("runs/deploy/detector_card"))
    args = parser.parse_args()
    every, per = parse_thresholds(args.threshold)
    sets = parse_sets(args.set)
    if args.source:
        sets["source"] = args.source
    m = stage(args.model, args.out, every, per, args.prompt, sets, args.replay_ann, args.replay_images,
              args.replay_limit, args.replay_category)
    shown = ("prompt", "input_size", "input_width", "input_height", "input_exponent", "output_exponent",
             "detect_threshold", "detect_threshold_source")
    always = set(shown) - {"input_size", "input_width", "input_height"}  # a model has one size form
    summary = {"card": m["card"], "prompt": m["config"]["prompt"],
               "models": [{k: x.get(k) for k in shown if k in x or k in always} for x in m["models"]]}
    if "replay" in m:
        summary["replay"] = {"files": len(m["replay"]["files"]), "skipped": len(m["replay"]["skipped"]),
                             "category_id": m["replay"]["category_id"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
