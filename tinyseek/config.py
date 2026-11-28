"""YAML experiment configs with single inheritance and dotted command-line overrides.

A config file may name a parent with ``base: other.yaml`` (resolved relative to the
file); the child is deep-merged over the parent. Overrides look like
``train.epochs=5`` and are parsed as YAML values, so ``5`` becomes an int and
``model.input_size=[256,192]`` a list (quote it in a shell: the brackets are glob characters).
"""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

Config = dict[str, Any]


def deep_merge(base: Config, override: Config) -> Config:
    """Return a new dict with ``override`` merged recursively over ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_with_base(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("base", None)
    if parent is None:
        return cfg
    return deep_merge(load_yaml_with_base(path.parent / parent), cfg)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    cfg = load_yaml_with_base(path)
    cfg = apply_overrides(cfg, overrides or [])
    cfg["_source"] = str(path)
    model = cfg.get("model")
    if isinstance(model, dict) and "input_size" in model:
        check_input_size(model["input_size"])  # fail here, not minutes into a run
    return cfg


# --- student input size ---------------------------------------------------------------------
#
# ``model.input_size`` is an int (square input, e.g. 192) or ``[width, height]`` (e.g.
# [256, 192] for the camera's 4:3 frame). Everything that needs the size goes through
# ``input_wh``; nothing else looks at the type.

# Both sides must be a multiple of this: the student's deepest feature map is at stride 32
# and the top-down fusion doubles it twice to meet the stride-16 and stride-8 maps
# (tinyseek.train.model), which only line up when the input divides by 32.
INPUT_MULTIPLE = 32


def input_wh(size: Any) -> tuple[int, int]:
    """``model.input_size`` (int, or [width, height]) -> (width, height)."""

    def is_side(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool) and v > 0

    wh = (size, size) if is_side(size) else size
    if not (isinstance(wh, (list, tuple)) and len(wh) == 2 and is_side(wh[0]) and is_side(wh[1])):
        raise ValueError(f"input size must be a positive int or [width, height], got {size!r}")
    return wh[0], wh[1]


def check_input_size(size: Any, multiple: int = INPUT_MULTIPLE) -> tuple[int, int]:
    """``input_wh`` plus the student's constraint: both sides a multiple of ``multiple``."""
    w, h = input_wh(size)
    if w % multiple or h % multiple:
        raise ValueError(
            f"model.input_size={size!r}: width and height must both be multiples of {multiple} "
            f"(the student fuses stride-8/16/32 features), e.g. 192, [256, 192] or [288, 224]")
    return w, h


def size_fields(size: Any) -> dict[str, int]:
    """The input-size entries of model_meta.json, stage.json and the SD-card sidecars.

    A square input keeps the one ``input_size`` key every earlier export and card has; a
    rectangular one is written as ``input_width`` and ``input_height`` and has no
    ``input_size``, so a reader that only knows squares fails instead of guessing.
    """
    w, h = input_wh(size)
    return {"input_size": w} if w == h else {"input_width": w, "input_height": h}


def fields_wh(meta: dict) -> tuple[int, int]:
    """(width, height) from a mapping written with ``size_fields`` (values may be strings)."""
    if "input_width" in meta and "input_height" in meta:
        return input_wh([int(meta["input_width"]), int(meta["input_height"])])
    if "input_size" in meta:
        return input_wh(int(meta["input_size"]))
    raise KeyError("no input_size or input_width/input_height")


def fill(template: str, cfg: Config) -> str:
    """Fill ``{key}`` placeholders in a path template from the config's top-level scalars."""
    scalars = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}
    return template.format(**scalars)


def load_prompts(path: str | Path = "configs/prompts.yaml") -> Config:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_run_config(cfg: Config, run_dir: str | Path) -> Path:
    """Write the resolved config plus the git revision next to a run's outputs."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = dict(cfg, _git=git_revision())
    out = run_dir / "config.yaml"
    with out.open("w") as f:
        yaml.safe_dump(json.loads(json.dumps(record, default=str)), f, sort_keys=False)
    return out


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
