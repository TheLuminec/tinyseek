"""Train a student detector for one prompt under one training condition.

    python -m tinyseek.train.train --config configs/real_only.yaml prompt=cup seed=1

The run directory (``runs/<prompt>/<condition>/seed<seed>`` unless ``run_name`` is
set) receives the resolved config, a JSONL training log, ``last.pt``, and
``eval.json``. The reported model is the final-epoch checkpoint: COCO val is the test
set, so it is never used to select checkpoints. ``train.eval_every`` evaluates
periodically for monitoring only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tinyseek.config import fill, load_config, load_prompts, save_run_config, seed_everything
from tinyseek.data.dataset import DetectionDataset, Record, load_coco_records
from tinyseek.eval.evaluate import evaluate_all
from tinyseek.train.loss import centernet_loss
from tinyseek.train.model import build_model


def load_sources(cfg: dict, category_id: int) -> list[Record]:
    """Concatenate all training sources; ``max_images`` on a source takes a seeded subset."""
    root = Path(cfg["data_root"])
    records: list[Record] = []
    for i, src in enumerate(cfg["train"]["sources"]):
        recs = load_coco_records(root / fill(src["ann"], cfg), root / fill(src["images"], cfg), category_id)
        limit = src.get("max_images")
        if limit is not None and limit < len(recs):
            rng = np.random.default_rng([cfg["seed"], i])
            recs = [recs[j] for j in sorted(rng.choice(len(recs), limit, replace=False))]
        print(f"source {i}: {len(recs)} images, {sum(len(r.boxes) for r in recs)} boxes ({src['ann']})")
        records += recs
    if not records:
        raise SystemExit("no training images; check train.sources")
    return records


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def run_dir_for(cfg: dict) -> Path:
    """runs/<prompt>/<condition>/seed<n>, or ``run_name`` with {placeholders} filled from the config."""
    name = cfg.get("run_name") or f"{cfg['prompt']}/{cfg.get('condition', 'custom')}/seed{cfg['seed']}"
    return Path(cfg["run_root"]) / fill(name, cfg)


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    run_dir = run_dir_for(cfg)
    save_run_config(cfg, run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg, mcfg = cfg["train"], cfg["model"]
    category_id = load_prompts()["prompts"][cfg["prompt"]]["coco_id"]

    records = load_sources(cfg, category_id)
    dataset = DetectionDataset(
        records, mcfg["input_size"], mcfg["output_stride"], train=True,
        augment_cfg=tcfg.get("augment"), seed=cfg["seed"],
    )
    loader = DataLoader(
        dataset, batch_size=tcfg["batch_size"], shuffle=True, drop_last=len(dataset) > tcfg["batch_size"],
        num_workers=tcfg["num_workers"], persistent_workers=False, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    model = build_model(mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(loader)
    warmup_steps = tcfg.get("warmup_epochs", 0) * len(loader)

    log = (run_dir / "train_log.jsonl").open("a")
    step = 0
    for epoch in range(tcfg["epochs"]):
        dataset.set_epoch(epoch)
        model.train()
        t0, running = time.time(), {}
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, warmup_steps, tcfg["lr"])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, parts = centernet_loss(model(batch["image"]), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            step += 1
        entry = {"epoch": epoch, "step": step, "time_s": round(time.time() - t0, 1)}
        entry.update({f"loss_{k}": v / len(loader) for k, v in running.items()})
        every = tcfg.get("eval_every", 0)
        if every and (epoch + 1) % every == 0 and epoch + 1 < tcfg["epochs"]:
            entry["eval"] = evaluate_all(model, cfg, device)
        print(json.dumps(entry), flush=True)
        log.write(json.dumps(entry) + "\n")
        log.flush()

    torch.save({"model": model.state_dict(), "epoch": tcfg["epochs"]}, run_dir / "last.pt")
    metrics = evaluate_all(model, cfg, device)
    (run_dir / "eval.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"final": metrics}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="e.g. prompt=backpack seed=2 train.epochs=5")
    args = parser.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
