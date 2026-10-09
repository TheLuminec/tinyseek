"""Tiny CenterNet-style detector: MobileNetV2 backbone, light top-down fusion, one stride-8 head.

Only ops that ESP-DL's quantizer handles are used: Conv, ReLU, Add, and nearest-neighbor
2x upsampling. The heatmap is returned as logits; sigmoid and peak picking happen in
post-processing (on device, only for cells above the logit threshold).
"""

from __future__ import annotations

import math

import timm
import torch
from torch import nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# timm feature indices for MobileNetV2: 2 -> stride 8, 3 -> stride 16, 4 -> stride 32.
_FEATURE_STRIDES = {2: 8, 3: 16, 4: 32}


def conv_bn_relu(cin: int, cout: int, k: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def sep_conv(c: int) -> nn.Sequential:
    """Depthwise 3x3 followed by pointwise 1x1."""
    return nn.Sequential(conv_bn_relu(c, c, 3, groups=c), conv_bn_relu(c, c, 1))


class TinyCenterNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        width_mult: float = 0.5,
        head_channels: int = 48,
        output_stride: int = 8,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if output_stride != 8:
            raise ValueError("only output_stride=8 is implemented")
        name = f"mobilenetv2_{round(width_mult * 100):03d}"
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=tuple(_FEATURE_STRIDES)
        )
        chans = self.backbone.feature_info.channels()
        self.lateral = nn.ModuleList(conv_bn_relu(c, head_channels) for c in chans)
        self.smooth = sep_conv(head_channels)
        self.heatmap = nn.Sequential(sep_conv(head_channels), nn.Conv2d(head_channels, num_classes, 1))
        self.size = nn.Sequential(sep_conv(head_channels), nn.Conv2d(head_channels, 2, 1))
        self.offset = nn.Sequential(sep_conv(head_channels), nn.Conv2d(head_channels, 2, 1))
        self.num_classes = num_classes
        self.stride = output_stride
        # Start with low objectness everywhere (prior 0.1) so the focal loss is stable early.
        nn.init.constant_(self.heatmap[-1].bias, -math.log((1 - 0.1) / 0.1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = [lat(f) for lat, f in zip(self.lateral, self.backbone(x))]
        y = feats[-1]
        for f in reversed(feats[:-1]):
            y = f + F.interpolate(y, scale_factor=2, mode="nearest")
        y = self.smooth(y)
        return {"heatmap": self.heatmap(y), "size": self.size(y), "offset": self.offset(y)}


class ExportWrapper(nn.Module):
    """Single-tensor output for deployment: channels = [heatmap logits..., size_w, size_h, off_x, off_y]."""

    def __init__(self, model: TinyCenterNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return torch.cat([out["heatmap"], out["size"], out["offset"]], dim=1)


def build_model(model_cfg: dict, num_classes: int = 1) -> TinyCenterNet:
    return TinyCenterNet(
        num_classes=num_classes,
        width_mult=model_cfg.get("width_mult", 0.5),
        head_channels=model_cfg.get("head_channels", 48),
        output_stride=model_cfg.get("output_stride", 8),
        pretrained=model_cfg.get("pretrained", True),
    )
