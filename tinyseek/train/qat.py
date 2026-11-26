"""Quantization-aware fine-tuning (QAT) of a trained student for ESP-DL int8.

    python -m tinyseek.train.qat --run runs/backpack/<condition>/seed0 --out runs/backpack/qat_<name>/seed0 \\
        [--config configs/qat.yaml] [qat.lr=3e-5 qat.rounds=2 ...]

Follows Espressif's own QAT recipe (esp-dl's YOLO11n-pose example): quantize with the
ESP-DL settings, then train the esp-ppq graph itself with the task loss through
``TorchExecutor.forward_with_gradient`` (a straight-through estimator on every fake-quant
node, scales held fixed). Here the graph is built exactly as ``tinyseek.train.export``
builds it (same calibration images, KL calibration, layerwise equalization, exponent-rule
int16 fallback), and after each round the trained weights are written back into the
PyTorch model, with every BatchNorm set to identity so the conv weights are the fused
ones. The next round re-quantizes from that model, so every round starts from exactly the
graph the unchanged export would build from those weights. Re-calibration moves a dozen or
more power-of-two activation exponents per round, which can undo the gain, so training
shifts random activation exponents by +-1 (``exp_jitter``) to make the weights work at
either neighbour, and the round whose export-equivalent graph scores best on the dev slice
is kept (``select_best``). The output is an ordinary run directory: config.yaml (the
float run's, plus a ``qat`` section), last.pt, train_log.jsonl, qat.json and, unless
``qat.final_eval`` is off, eval.json; export it with ``tinyseek.train.export --run <out>``.

Method choices are made on a dev slice: ``qat.dev_images`` images of the run's first
training source, held out of QAT and scored with their teacher labels, so the camera
test sets are not used for tuning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tinyseek.config import fill, load_config, load_prompts, save_run_config, seed_everything
from tinyseek.data.dataset import DetectionDataset, Record, collate_eval
from tinyseek.eval.evaluate import evaluate_all, predict
from tinyseek.eval.metrics import coco_ap50
from tinyseek.train.loss import centernet_loss
from tinyseek.train.model import ExportWrapper, build_model
from tinyseek.train.train import load_sources, lr_at

QAT_DEFAULTS: dict = {
    "rounds": 2,             # re-quantize (as export does) at the start of each round
    "epochs_per_round": 3,
    "lr": 3.0e-5,
    "warmup_steps": 10,
    "batch_size": 32,
    "num_workers": 8,
    "num_threads": 8,
    "train_heads_only": False,  # True: only convs outside the backbone are trained
    "calib_images": 512,     # must match the export's --calib-images
    "target": "esp32s3",
    "dev_images": 200,
    "dev_seed": 0,           # fixed, so every seed of a condition shares one dev slice
    "final_eval": True,      # eval.json on the run's eval sets, as train.py writes
    "select_best": True,     # keep the round whose export-equivalent int8 graph scores best on dev
    "exp_jitter": 0.0,       # per step, chance that each activation exponent is shifted by +-1
}


# ---------------------------------------------------------------- graph <-> model mapping

def module_path(op_name: str) -> str:
    """ONNX op name from the exported ``ExportWrapper`` -> TinyCenterNet module path.

    torch.onnx scopes repeat a container's name in its children ("heatmap/heatmap.0/
    heatmap.0.0"), so a segment that extends the previous one replaces it:
    ``/model/backbone/blocks.0/blocks.0.0/conv_dw/Conv`` -> ``backbone.blocks.0.0.conv_dw``.
    """
    parts: list[str] = []
    for seg in op_name.strip("/").split("/")[:-1]:
        if parts and seg.startswith(parts[-1] + "."):
            parts[-1] = seg
        else:
            parts.append(seg)
    if not parts or parts[0] != "model":
        raise ValueError(f"not an op of ExportWrapper(model): {op_name!r}")
    return ".".join(parts[1:])


def conv_bn_pairs(model: nn.Module, input_size: int) -> dict[str, str | None]:
    """Every Conv2d's name -> the BatchNorm2d that consumes its output directly (or None)."""
    produced: dict[int, str] = {}
    pairs: dict[str, str | None] = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            def conv_hook(m, inp, out, name=name):
                produced[id(out)] = name
                pairs.setdefault(name, None)
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.BatchNorm2d):
            def bn_hook(m, inp, name=name):
                src = produced.get(id(inp[0]))
                if src is not None:
                    pairs[src] = name
            hooks.append(mod.register_forward_pre_hook(bn_hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, input_size, input_size))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return pairs


@torch.no_grad()
def write_back(model: nn.Module, fused: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
               pairs: dict[str, str | None]) -> None:
    """Load fused conv weights/biases into ``model``; each conv's BatchNorm becomes identity.

    ``fused`` maps conv module names to (weight, bias) as the ONNX/esp-ppq graph holds them
    (BatchNorm folded in, possibly equalized). Every conv in ``pairs`` must be present.
    """
    modules = dict(model.named_modules())
    missing = sorted(set(pairs) - set(fused))
    if missing:
        raise KeyError(f"graph has no weights for {missing}")
    for name, (weight, bias) in fused.items():
        conv = modules[name]
        if conv.weight.shape != weight.shape:
            raise ValueError(f"{name}: shape {tuple(weight.shape)} != {tuple(conv.weight.shape)}")
        conv.weight.copy_(weight)
        bias = torch.zeros(weight.shape[0]) if bias is None else bias
        bn_name = pairs.get(name)
        if bn_name is None:
            if conv.bias is None:
                raise ValueError(f"{name} has a fused bias but no BatchNorm or bias to hold it")
            conv.bias.copy_(bias)
            continue
        if conv.bias is not None:
            conv.bias.zero_()
        bn = modules[bn_name]
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0 - bn.eps)  # sqrt(var + eps) == 1
        bn.weight.fill_(1.0)
        bn.bias.copy_(bias)


def graph_convs(graph) -> dict[str, object]:
    """Module path -> esp-ppq Conv operation, for every Conv in the graph."""
    return {module_path(op.name): op for op in graph.operations.values() if op.type == "Conv"}


def fused_from_graph(graph) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    out = {}
    for name, op in graph_convs(graph).items():
        w = op.inputs[1].value.detach().clone().float()
        b = op.inputs[2].value.detach().clone().float() if len(op.inputs) > 2 else None
        out[name] = (w, b)
    return out


# ---------------------------------------------------------------- export-identical quantization

def calibration_records(cfg: dict, records: list[Record], calib_images: int) -> list[Record]:
    """The same images tinyseek.train.export draws for calibration."""
    rng = np.random.default_rng(cfg["seed"])
    return [records[i] for i in rng.choice(len(records), min(calib_images, len(records)), replace=False)]


def quantize_like_export(model: nn.Module, calib: list[Record], size: int, target: str, scratch: Path):
    """esp-ppq int8 graph built as tinyseek.train.export builds it; returns (graph, int16 ops)."""
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_torch, get_target_platform

    from tinyseek.train.export import exponent_violations

    scratch.mkdir(parents=True, exist_ok=True)
    model.eval()
    loader = DataLoader(DetectionDataset(calib, size, model.stride, train=False), batch_size=1,
                        collate_fn=collate_eval)

    def quantize(int16_ops: list[str]):
        setting = QuantizationSettingFactory.espdl_setting(8)
        setting.equalization = True
        return espdl_quantize_torch(
            model=ExportWrapper(model), espdl_export_file=str(scratch / "model.espdl"),
            calib_dataloader=loader, calib_steps=len(calib), input_shape=[1, 3, size, size],
            target=target, num_of_bits=8, collate_fn=lambda batch: batch["image"],
            dispatching_override={op: get_target_platform(target, "w16a16") for op in int16_ops},
            setting=setting, device="cpu", error_report=False, verbose=0, skip_export=True,
        )

    int16_ops: list[str] = []
    graph = quantize(int16_ops)
    for _ in range(4):
        bad = [op for op in exponent_violations(graph) if op not in int16_ops]
        if not bad:
            return graph, int16_ops
        int16_ops += bad
        graph = quantize(int16_ops)
    raise RuntimeError(f"ESP-DL exponent constraint still violated by {bad}")


def exponents(graph) -> dict[str, list[int]]:
    """Per-op power-of-two exponents of the inputs and outputs (to see what re-calibration moved)."""
    from esp_ppq.IR import QuantableOperation

    def e(cfg) -> int:
        return round(float(torch.log2(torch.as_tensor(cfg.scale).flatten()[0])))

    out = {}
    for op in graph.topological_sort():
        if isinstance(op, QuantableOperation):
            cfgs = list(op.config.input_quantization_config[:2]) + list(op.config.output_quantization_config)
            out[op.name] = [e(c) for c in cfgs if c.scale is not None]
    return out


class GraphModel:
    """An esp-ppq graph behind TinyCenterNet's interface (dict outputs, ``stride``), batched."""

    def __init__(self, executor, num_classes: int, stride: int, grad: bool = False) -> None:
        self.executor, self.num_classes, self.stride, self.grad = executor, num_classes, stride, grad

    def eval(self) -> GraphModel:
        return self

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        run = self.executor.forward_with_gradient if self.grad else self.executor.forward
        out = run(x)[0]
        return split_output(out, self.num_classes)


def split_output(out: torch.Tensor, num_classes: int) -> dict[str, torch.Tensor]:
    c = num_classes
    return {"heatmap": out[:, :c], "size": out[:, c:c + 2], "offset": out[:, c + 2:c + 4]}


def trainable_tensors(graph, heads_only: bool = False) -> list[torch.Tensor]:
    """Conv weights and biases of the graph (not Clip bounds or Resize scales)."""
    params = []
    for name, op in graph_convs(graph).items():
        if heads_only and name.startswith("backbone."):
            continue
        params += [v.value for v in op.inputs[1:] if v.is_parameter]
    return params


def activation_configs(graph) -> list:
    """The independent activation quantization configs (each ACTIVATED root config once).

    Followers (aligned Add/Concat inputs, fused Conv outputs) share their root's scale, so
    changing a root changes the whole group. The graph input's config is not included: it
    comes from the images alone and cannot move when the weights change.
    """
    from esp_ppq.core import QuantizationStates
    from esp_ppq.IR import QuantableOperation

    out, seen = [], set()
    for op in graph.topological_sort():
        if not isinstance(op, QuantableOperation):
            continue
        for cfg, var in zip(op.config.output_quantization_config, op.outputs):
            if var.is_parameter or id(cfg) in seen:
                continue
            if cfg.dominated_by is cfg and cfg.state == QuantizationStates.ACTIVATED:
                seen.add(id(cfg))
                out.append(cfg)
    return out


class ExponentJitter:
    """Shift a random subset of activation exponents by +-1 for one training step.

    Export re-calibrates every activation scale from the trained weights, and a power-of-two
    scale near a rounding boundary flips by a factor of 2 (seen in practice: a dozen or more
    exponents per round). Training under random +-1 shifts makes the weights work at either
    neighbouring exponent, so the re-calibrated graph keeps the QAT gain.
    """

    def __init__(self, configs: list, prob: float, rng: np.random.Generator) -> None:
        self.configs, self.prob, self.rng = configs, prob, rng
        self.saved: list[tuple[object, torch.Tensor]] = []

    def __enter__(self) -> list[tuple[object, torch.Tensor]]:
        """Apply the shifts; returns the (config, original scale) pairs that were changed."""
        for cfg in self.configs:
            if self.prob > 0 and self.rng.random() < self.prob:
                self.saved.append((cfg, cfg.scale))
                cfg.scale = cfg.scale * (2.0 if self.rng.random() < 0.5 else 0.5)
        return list(self.saved)

    def __exit__(self, *exc) -> None:
        for cfg, scale in reversed(self.saved):
            cfg.scale = scale
        self.saved = []


# ---------------------------------------------------------------- data and scoring

def split_dev(records: list[Record], dev_images: int, seed: int) -> tuple[list[Record], list[Record]]:
    """Seeded split into (QAT training, dev) records; order within each part is preserved."""
    if not 0 <= dev_images < len(records):
        raise ValueError(f"dev_images must be in [0, {len(records)}), got {dev_images}")
    dev_idx = set(np.random.default_rng(seed).choice(len(records), dev_images, replace=False).tolist())
    train = [r for i, r in enumerate(records) if i not in dev_idx]
    dev = [r for i, r in enumerate(records) if i in dev_idx]
    return train, dev


def dev_ap(model, dev: list[Record], ann: str, category_id: int, size: int, ev: dict) -> float:
    with torch.no_grad():
        dets = predict(model, dev, size, torch.device("cpu"), ev["score_threshold"], ev["max_detections"],
                       batch_size=16, num_workers=4)
    return coco_ap50(ann, dets, category_id, image_ids=sorted(r.image_id for r in dev))


def exponent_moves(before: dict[str, list[int]], after: dict[str, list[int]]) -> dict[str, list[list[int]]]:
    """Ops whose exponents differ between two graphs: name -> [before, after]."""
    return {k: [before[k], after[k]] for k in sorted(after) if k in before and before[k] != after[k]}


def check_write_back(model: nn.Module, graph, pairs: dict[str, str | None], mcfg: dict, size: int) -> None:
    """Raise unless writing the graph's float weights into a copy of ``model`` keeps its output."""
    check = build_model(dict(mcfg, pretrained=False))
    check.load_state_dict(model.state_dict())
    write_back(check, fused_from_graph(graph), pairs)
    check.eval()
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        err = (ExportWrapper(check)(x) - ExportWrapper(model)(x)).abs().max().item()
    if err > 1e-3:
        raise RuntimeError(f"graph -> model write-back changes the float output by {err:.2e}")


# ---------------------------------------------------------------- training

def resolve_settings(qcfg: dict) -> dict:
    """QAT_DEFAULTS overridden by ``qcfg``, each value coerced to its default's type.

    YAML 1.1 reads ``1e-4`` (no dot) as a string, so numbers are converted explicitly.
    """
    unknown = sorted(set(qcfg) - set(QAT_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown qat settings: {unknown}")
    out = dict(QAT_DEFAULTS)
    for key, value in qcfg.items():
        kind = type(QAT_DEFAULTS[key])
        if kind is bool and isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on")
        out[key] = kind(value)
    return out


def qat(run: Path, out: Path, qcfg: dict) -> dict:
    """Fine-tune ``run``'s float model through its ESP-DL int8 graph; write a run dir to ``out``."""
    from esp_ppq.executor import TorchExecutor
    from esp_ppq.IR import TrainableGraph

    q = resolve_settings(qcfg)
    torch.set_num_threads(q["num_threads"])
    cfg = load_config(run / "config.yaml")
    cfg.pop("_git", None)
    cfg["qat"], cfg["qat_source"] = q, str(run)
    seed_everything(cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, out)
    scratch = out / "qat_scratch"

    mcfg, size = cfg["model"], cfg["model"]["input_size"]
    model = build_model(dict(mcfg, pretrained=False))
    ckpt = torch.load(run / "last.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    reference = build_model(dict(mcfg, pretrained=False))
    reference.load_state_dict(ckpt["model"])
    reference.eval()
    pairs = conv_bn_pairs(model, size)

    category_id = load_prompts()["prompts"][cfg["prompt"]]["coco_id"]
    records = load_sources(cfg, category_id)
    calib = calibration_records(cfg, records, q["calib_images"])
    root = Path(cfg["data_root"])
    src0 = cfg["train"]["sources"][0]
    first = [r for r in records if str(r.image_path).startswith(str(root / fill(src0["images"], cfg)))]
    _, dev = split_dev(first, q["dev_images"], q["dev_seed"]) if q["dev_images"] else ([], [])
    dev_ids = {(str(r.image_path), r.image_id) for r in dev}
    train_recs = [r for r in records if (str(r.image_path), r.image_id) not in dev_ids]
    dev_ann = str(root / fill(src0["ann"], cfg))
    ev = cfg["eval"]
    print(f"QAT on {len(train_recs)} images, dev slice {len(dev)} images", flush=True)

    dataset = DetectionDataset(train_recs, size, model.stride, train=True,
                               augment_cfg=cfg["train"].get("augment"), seed=cfg["seed"])
    loader = DataLoader(dataset, batch_size=q["batch_size"], shuffle=True, drop_last=True,
                        num_workers=q["num_workers"], generator=torch.Generator().manual_seed(cfg["seed"]))
    total = q["rounds"] * q["epochs_per_round"] * len(loader)

    summary: dict = {"source_run": str(run), "qat": q, "train_images": len(train_recs), "dev_images": len(dev),
                     "rounds": []}
    if dev:
        summary["dev_ap50_float_source"] = dev_ap(reference, dev, dev_ann, category_id, size, ev)
    log = (out / "train_log.jsonl").open("a")
    step, epoch, t_start = 0, 0, time.time()
    prev_exp: dict | None = None
    best: dict | None = None
    jitter_rng = np.random.default_rng([cfg["seed"], 7])
    for rnd in range(q["rounds"] + 1):
        t0 = time.time()
        graph, int16_ops = quantize_like_export(model, calib, size, q["target"], scratch)
        exps = exponents(graph)
        moved = exponent_moves(prev_exp, exps) if prev_exp is not None else {}
        prev_exp = exps
        if rnd == 0:  # the mapping must reproduce the float function exactly before training starts
            check_write_back(model, graph, pairs, mcfg, size)
        executor = TorchExecutor(graph, device="cpu")
        # Every round starts from a graph built exactly as export builds it from the current weights,
        # so its dev score is the score this checkpoint will have after export.
        entry = {"round": rnd, "int16_layers": int16_ops, "exponents_moved": moved,
                 "quantize_s": round(time.time() - t0, 1)}
        if dev:
            entry["dev_ap50_int8"] = dev_ap(GraphModel(executor, model.num_classes, model.stride), dev,
                                            dev_ann, category_id, size, ev)
            entry["dev_ap50_float"] = dev_ap(model, dev, dev_ann, category_id, size, ev)
        print(json.dumps(entry), flush=True)
        log.write(json.dumps(entry) + "\n")
        summary["rounds"].append(entry)
        score = entry.get("dev_ap50_int8", 0.0)
        if best is None or not q["select_best"] or score > best["score"]:
            best = {"round": rnd, "score": score,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if rnd == q["rounds"]:
            break

        params = trainable_tensors(graph, q["train_heads_only"])
        for p in TrainableGraph(graph).parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True
        optimizer = torch.optim.Adam(params, lr=q["lr"])
        net = GraphModel(executor, model.num_classes, model.stride, grad=True)
        act_cfgs = activation_configs(graph)
        for _ in range(q["epochs_per_round"]):
            dataset.set_epoch(1000 + epoch)  # augmentations unlike any the float run drew
            running, te = {}, time.time()
            for batch in loader:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, total, q["warmup_steps"], q["lr"])
                with ExponentJitter(act_cfgs, q["exp_jitter"], jitter_rng):
                    loss, parts = centernet_loss(net(batch["image"]), batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                optimizer.step()
                for k, v in parts.items():
                    running[k] = running.get(k, 0.0) + v
                step += 1
            row = {"round": rnd, "epoch": epoch, "step": step, "time_s": round(time.time() - te, 1)}
            row.update({f"loss_{k}": v / len(loader) for k, v in running.items()})
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            epoch += 1
        for p in params:
            p.requires_grad = False
        if dev:  # the trained graph before re-calibration; the next round shows what re-calibration did
            entry["dev_ap50_int8_trained"] = dev_ap(GraphModel(executor, model.num_classes, model.stride),
                                                    dev, dev_ann, category_id, size, ev)
            print(json.dumps({"round": rnd, "dev_ap50_int8_trained": entry["dev_ap50_int8_trained"]}),
                  flush=True)
        write_back(model, fused_from_graph(graph), pairs)
        model.eval()

    model.load_state_dict(best["state"])
    summary["selected_round"] = best["round"]
    summary["final"] = next(e for e in summary["rounds"] if e["round"] == best["round"])
    summary["train_s"] = round(time.time() - t_start)
    torch.save({"model": model.state_dict(), "epoch": ckpt.get("epoch"), "qat_epochs": epoch,
                "qat_round": best["round"]}, out / "last.pt")
    (out / "qat.json").write_text(json.dumps(summary, indent=2))
    shutil.rmtree(scratch, ignore_errors=True)  # per-round ONNX/espdl intermediates
    print(json.dumps({"selected_round": best["round"], "dev_ap50_int8": best["score"]}), flush=True)
    if q["final_eval"]:
        metrics = evaluate_all(model, cfg, torch.device("cpu"))
        (out / "eval.json").write_text(json.dumps(metrics, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="float run directory (config.yaml, last.pt)")
    parser.add_argument("--out", required=True, type=Path, help="new run directory for the QAT model")
    parser.add_argument("--config", default="configs/qat.yaml", help="YAML with a `qat:` section")
    parser.add_argument("overrides", nargs="*", help="e.g. qat.lr=1e-4 qat.rounds=3")
    args = parser.parse_args()
    if args.out.resolve() == args.run.resolve():
        raise SystemExit("--out must differ from --run (the float run is kept as it is)")
    qcfg = load_config(args.config, args.overrides).get("qat", {}) if args.config else {}
    try:
        resolve_settings(qcfg)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(json.dumps(qat(args.run, args.out, qcfg), indent=2))


if __name__ == "__main__":
    main()
