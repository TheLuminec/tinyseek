"""YAML experiment configs with single inheritance and dotted command-line overrides.

A config file may name a parent with ``base: other.yaml`` (resolved relative to the
file); the child is deep-merged over the parent. Overrides look like
``train.epochs=5`` and are parsed as YAML scalars, so ``5`` becomes an int.
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
    return cfg


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
