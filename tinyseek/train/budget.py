"""Check a student config against the ESP32-S3 memory and compute budget.

    python -m tinyseek.train.budget --config configs/base.yaml [model.input_size=160]

Estimates are for int8 weights and activations. Peak activation memory is
approximated as the largest (input + output) pair of any single layer, which is
what a sequential executor reusing buffers must hold at once; the top-down fusion
also keeps the stride-16/32 laterals alive, so a margin is added on top.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn

from tinyseek.config import load_config
from tinyseek.train.model import build_model

# Freenove/generic ESP32-S3 N16R8: 8 MB PSRAM, 16 MB flash. Leave room for the camera
# frame buffers (2x QVGA RGB565 = 300 KB), Wi-Fi, and the display/stream path.
PSRAM_BYTES = 8 * 1024 * 1024
ACTIVATION_BUDGET = 2 * 1024 * 1024
WEIGHT_BUDGET = 3 * 1024 * 1024
MACS_BUDGET = 150e6  # very roughly ~100-200 ms on the S3 with ESP-DL int8 kernels


@dataclass
class Budget:
    input_size: int
    params: int
    weight_bytes_int8: int
    peak_activation_bytes_int8: int
    macs: int
    fits: bool


def measure(model: nn.Module, input_size: int) -> Budget:
    peak = 0
    macs = 0

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal peak, macs
        if not isinstance(output, torch.Tensor):
            return
        live = sum(t.numel() for t in inputs if isinstance(t, torch.Tensor)) + output.numel()
        peak = max(peak, live)
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs += output.numel() * k

    leaves = [m for m in model.modules() if not list(m.children())]
    handles = [m.register_forward_hook(hook) for m in leaves]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))
    for h in handles:
        h.remove()

    params = sum(p.numel() for p in model.parameters())
    # BatchNorm folds into the preceding conv at export, so its parameters don't ship.
    shipped = params - sum(p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters())
    activations = int(peak * 1.5)
    return Budget(
        input_size=input_size,
        params=params,
        weight_bytes_int8=shipped,
        peak_activation_bytes_int8=activations,
        macs=macs,
        fits=shipped <= WEIGHT_BUDGET and activations <= ACTIVATION_BUDGET and macs <= MACS_BUDGET,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    model = build_model(dict(cfg["model"], pretrained=False))
    budget = measure(model, cfg["model"]["input_size"])
    print(json.dumps(asdict(budget), indent=2))
    if not budget.fits:
        raise SystemExit("model exceeds the ESP32-S3 budget")


if __name__ == "__main__":
    main()
