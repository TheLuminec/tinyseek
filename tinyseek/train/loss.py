"""CenterNet losses: penalty-reduced focal loss on the heatmap, L1 on size and offset at centers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Sum of the CenterNet focal loss, normalized by the number of positive cells."""
    pred = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg_weight = (1 - target).pow(beta)
    pos_loss = -F.logsigmoid(logits) * (1 - pred).pow(alpha) * pos
    neg_loss = -F.logsigmoid(-logits) * pred.pow(alpha) * neg_weight * (1 - pos)
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def gather_at(feature: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) feature, (B, K) flat cell indices -> (B, K, C)."""
    b, c, h, w = feature.shape
    flat = feature.view(b, c, h * w)
    return flat.gather(2, index.unsqueeze(1).expand(b, c, -1)).permute(0, 2, 1)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    return (F.l1_loss(pred * m, target * m, reduction="sum")) / (m.sum() * pred.shape[-1]).clamp(min=1.0)


def centernet_loss(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    size_weight: float = 1.0,
    offset_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    hm = focal_loss(out["heatmap"], targets["heatmap"])
    size = masked_l1(gather_at(out["size"], targets["index"]), targets["size"], targets["mask"])
    off = masked_l1(gather_at(out["offset"], targets["index"]), targets["offset"], targets["mask"])
    total = hm + size_weight * size + offset_weight * off
    return total, {"heatmap": hm.item(), "size": size.item(), "offset": off.item()}
