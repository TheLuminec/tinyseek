"""Quantize a trained student to int8 and export it for ESP-DL (``.espdl``).

    python -m tinyseek.train.export --run runs/cup/real_only/seed0 [--calib-images 512] [--eval-images 1000]

Calibration uses letterboxed (un-augmented) images from the run's own training
sources. Writes ``model.espdl`` and ``model_meta.json`` (input size, normalization,
output channel layout) into the run directory. With ``--eval-images`` the simulated
int8 graph is scored on the first eval set so the quantization drop is measured
with the same metric as the float model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tinyseek.config import fill, load_config, load_prompts
from tinyseek.data.dataset import DetectionDataset, collate_eval, load_coco_records
from tinyseek.eval.evaluate import predict
from tinyseek.eval.metrics import coco_ap50
from tinyseek.train.model import IMAGENET_MEAN, IMAGENET_STD, ExportWrapper, build_model
from tinyseek.train.train import load_sources


class QuantizedAdapter:
    """Makes an esp-ppq simulated int8 graph look like a TinyCenterNet for ``predict``."""

    def __init__(self, executor, num_classes: int, stride: int) -> None:
        self.executor = executor
        self.num_classes = num_classes
        self.stride = stride

    def eval(self) -> QuantizedAdapter:
        return self

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Exported with batch size 1, so run image by image.
        out = torch.cat([self.executor(xi.unsqueeze(0))[0] for xi in x])
        c = self.num_classes
        return {"heatmap": out[:, :c], "size": out[:, c : c + 2], "offset": out[:, c + 2 : c + 4]}


def exponent_violations(graph) -> list[str]:
    """Ops ESP-DL would compute wrongly: it needs ``out_exp - in_exp - weight_exp >= 0``.

    ESP-DL quantizes per tensor with power-of-two scales (``scale = 2**exp``) and can only
    right-shift the accumulator. Mirrors the check esp-ppq logs (but does not raise on).
    """
    from esp_ppq.IR import QuantableOperation

    def exp(cfg) -> float:
        return float(torch.log2(torch.as_tensor(cfg.scale).flatten()[0]))

    bad = []
    for op in graph.topological_sort():
        if op.type in ("Conv", "Gemm", "MatMul", "Mul") and isinstance(op, QuantableOperation):
            ins, outs = op.config.input_quantization_config, op.config.output_quantization_config
            if len(ins) >= 2 and exp(outs[0]) - exp(ins[0]) - exp(ins[1]) < 0:
                bad.append(op.name)
    return bad


def export(run: Path, calib_images: int, eval_images: int, target: str) -> dict:
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_torch, get_target_platform
    from esp_ppq.executor import TorchExecutor

    cfg = load_config(run / "config.yaml")
    mcfg = cfg["model"]
    size = mcfg["input_size"]
    model = build_model(dict(mcfg, pretrained=False))
    model.load_state_dict(torch.load(run / "last.pt", map_location="cpu")["model"])
    model.eval()

    category_id = load_prompts()["prompts"][cfg["prompt"]]["coco_id"]
    records = load_sources(cfg, category_id)
    rng = np.random.default_rng(cfg["seed"])
    calib = [records[i] for i in rng.choice(len(records), min(calib_images, len(records)), replace=False)]
    calib_loader = DataLoader(
        DetectionDataset(calib, size, model.stride, train=False), batch_size=1, collate_fn=collate_eval
    )

    out_file = run / "model.espdl"

    def quantize(int16_ops: list[str]):
        setting = QuantizationSettingFactory.espdl_setting()
        setting.equalization = True
        return espdl_quantize_torch(
            model=ExportWrapper(model),
            espdl_export_file=str(out_file),
            calib_dataloader=calib_loader,
            calib_steps=len(calib),
            input_shape=[1, 3, size, size],
            target=target,
            num_of_bits=8,
            collate_fn=lambda batch: batch["image"],
            dispatching_override={op: get_target_platform(target, "w16a16") for op in int16_ops},
            setting=setting,
            device="cpu",
            error_report=False,
            verbose=0,
        )

    graph = quantize([])
    int16_ops = exponent_violations(graph)
    if int16_ops:
        print(f"re-quantizing with {len(int16_ops)} layer(s) in int16: {int16_ops}")
        graph = quantize(int16_ops)
        remaining = exponent_violations(graph)
        if remaining:
            raise SystemExit(f"ESP-DL exponent constraint still violated by {remaining}")
    meta = {
        "prompt": cfg["prompt"],
        "input_size": size,
        "input_layout": "NCHW RGB, letterboxed with pad value 114",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "stride": model.stride,
        "num_classes": model.num_classes,
        "output_channels": ["heatmap_logit"] * model.num_classes + ["log_w", "log_h", "off_x", "off_y"],
        "target": target,
    }
    (run / "model_meta.json").write_text(json.dumps(meta, indent=2))
    result = {"espdl_bytes": out_file.stat().st_size, "int16_layers": int16_ops}

    if eval_images:
        name, spec = next(iter(cfg["eval"]["sets"].items()))
        root = Path(cfg["data_root"])
        ann = str(root / fill(spec["ann"], cfg))
        recs = load_coco_records(ann, root / fill(spec["images"], cfg), category_id)[:eval_images]
        ids = {r.image_id for r in recs}
        ev = cfg["eval"]
        int8 = QuantizedAdapter(TorchExecutor(graph, device="cpu"), model.num_classes, model.stride)

        def ap(m: nn.Module | QuantizedAdapter) -> float:
            dets = predict(m, recs, size, torch.device("cpu"), ev["score_threshold"], ev["max_detections"],
                           batch_size=16, num_workers=4)
            return coco_ap50(ann, dets, category_id, image_ids=sorted(ids))

        result.update({"eval_set": name, "eval_images": len(recs), "ap50_float": ap(model), "ap50_int8": ap(int8)})
    (run / "export.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--calib-images", type=int, default=512)
    parser.add_argument("--eval-images", type=int, default=1000, help="0 to skip the int8 accuracy check")
    parser.add_argument("--target", default="esp32s3")
    args = parser.parse_args()
    print(json.dumps(export(Path(args.run), args.calib_images, args.eval_images, args.target), indent=2))


if __name__ == "__main__":
    main()
