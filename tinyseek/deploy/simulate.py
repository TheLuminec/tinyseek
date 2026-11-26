"""esp-ppq's simulated int8 graph for an exported run: the PC's expected board output.

``tinyseek.train.export`` does not save the quantized graph, only ``model.espdl``. This
rebuilds it with the same calibration (same records, same seed, same settings, same
int16 fallback) but writes the ``.espdl`` to a scratch directory, never into the run,
and checks that the rebuilt model has the same ops, weights and exponents as the
deployed one (``same_model``). Only then is the simulation a valid reference for that
file. CPU only; about 2 minutes per model on a desktop CPU.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tinyseek.config import load_config, load_prompts
from tinyseek.data.dataset import DetectionDataset, collate_eval
from tinyseek.deploy.reference import dequantize, quantize
from tinyseek.train.export import exponent_violations
from tinyseek.train.model import ExportWrapper, build_model
from tinyseek.train.train import load_sources


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info_signature(text: str) -> tuple[list[str], dict[str, str]]:
    """Order-independent content of an esp-ppq ``model.info`` dump: graph lines and tensors.

    Re-exporting the same graph does not give a byte-identical ``.espdl`` (initializers
    are serialized in set order), but the dump lists every op, every initializer's values
    and every tensor's exponent, so equal signatures mean the same deployed model.
    """
    marker = "initializers value:"
    text = _canonical_auto_names(text)
    head, _, values = text.partition(marker)
    graph = sorted(line.strip() for line in head.splitlines() if line.strip())
    tensors = {}
    for block in re.split(r"\n(?=%)", values)[1:]:
        tensors[block.split(",")[0]] = re.sub(r"\s+", " ", block).strip()
    return graph, tensors


_AUTO_NAME = re.compile(r"%PPQ_Variable_\d+\b")


def _canonical_auto_names(text: str) -> str:
    """Rename esp-ppq's generated ``PPQ_Variable_N`` tensors by first use in the graph body.

    The numbering is not stable between exports (e.g. a Clip's min and max scalars swap
    numbers), while the op order in the body is, so first-use order gives the same name
    to the same tensor in both dumps.
    """
    body = text.partition(") {")[2].partition("\n}")[0]
    mapping: dict[str, str] = {}
    for name in _AUTO_NAME.findall(body):
        mapping.setdefault(name, f"%auto_{len(mapping)}")
    return _AUTO_NAME.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def same_model(info_a: Path, info_b: Path) -> bool:
    return info_signature(Path(info_a).read_text()) == info_signature(Path(info_b).read_text())


def _exponent(cfg) -> int:
    return round(float(torch.log2(torch.as_tensor(cfg.scale).flatten()[0])))


@dataclass
class SimulatedModel:
    executor: object
    input_exponent: int
    output_exponent: int
    input_size: int
    num_classes: int
    stride: int
    int16_ops: list[str]
    identical_to_deployed: bool

    def run(self, q_in: np.ndarray) -> np.ndarray:
        """(S, S, 3) int8 HWC input -> (G, G, C) int8 NHWC output, as the board should produce."""
        x = torch.from_numpy(dequantize(q_in, self.input_exponent).transpose(2, 0, 1).copy())[None]
        out = self.executor(x)[0][0].detach().numpy()  # (C, G, G), already fake-quantized
        return quantize(out.transpose(1, 2, 0), self.output_exponent)


def rebuild(run: Path, scratch: Path, calib_images: int = 512, target: str = "esp32s3") -> SimulatedModel:
    """Re-run export's quantization for ``run`` into ``scratch`` and return its executor.

    Kept in step with ``tinyseek.train.export.export`` by hand (that function writes into
    the run directory, which this tool must not touch); the model.info comparison below
    is what catches drift.
    """
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_torch, get_target_platform
    from esp_ppq.executor import TorchExecutor

    run, scratch = Path(run), Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
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
    loader = DataLoader(DetectionDataset(calib, size, model.stride, train=False), batch_size=1, collate_fn=collate_eval)
    out_file = scratch / "model.espdl"

    # The export's recipe (tinyseek.train.export writes it to export.json): bit width and the
    # layers it ran in int16. Older exports have neither and were plain w8a8.
    recipe = json.loads((run / "export.json").read_text()) if (run / "export.json").exists() else {}
    bits = 16 if recipe.get("quant") == "w16a16" else 8

    def quantize_graph(int16_ops: list[str]):
        setting = QuantizationSettingFactory.espdl_setting(bits)
        setting.equalization = True
        return espdl_quantize_torch(
            model=ExportWrapper(model), espdl_export_file=str(out_file), calib_dataloader=loader,
            calib_steps=len(calib), input_shape=[1, 3, size, size], target=target, num_of_bits=bits,
            collate_fn=lambda batch: batch["image"],
            dispatching_override={op: get_target_platform(target, "w16a16") for op in int16_ops},
            setting=setting, device="cpu", error_report=False, verbose=0,
        )

    int16_ops = list(recipe.get("int16_layers", [])) if bits == 8 else []
    graph = quantize_graph(int16_ops)
    for _ in range(4):  # as export does: add any layer that now breaks the exponent rule
        bad = [op for op in exponent_violations(graph) if op not in int16_ops] if bits == 8 else []
        if not bad:
            break
        int16_ops += bad
        graph = quantize_graph(int16_ops)
    in_var = graph.inputs[next(iter(graph.inputs))]
    out_var = graph.outputs[next(iter(graph.outputs))]
    in_cfg = in_var.dest_ops[0].config.input_quantization_config[0]
    out_cfg = out_var.source_op.config.output_quantization_config[0]
    return SimulatedModel(
        executor=TorchExecutor(graph, device="cpu"),
        input_exponent=_exponent(in_cfg),
        output_exponent=_exponent(out_cfg),
        input_size=size,
        num_classes=model.num_classes,
        stride=model.stride,
        int16_ops=int16_ops,
        identical_to_deployed=same_model(scratch / "model.info", run / "model.info"),
    )
