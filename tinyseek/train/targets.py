"""CenterNet-style target encoding and decoding on a single output stride.

Each object is a Gaussian peak on a per-class heatmap at its center cell, plus a
sub-cell center offset and a log-scale box size (in grid units) at that cell.
Decoding mirrors what the firmware does: 3x3 local maxima, top-k, then
``center = (cell + offset) * stride`` and ``wh = exp(size) * stride``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MAX_OBJECTS = 32


def gaussian_sigmas(w: float, h: float, alpha: float = 0.54) -> tuple[float, float]:
    """Box-proportional elliptical Gaussian radii in grid units (TTFNet), floored for tiny boxes."""
    return max(alpha * w / 6.0, 0.4), max(alpha * h / 6.0, 0.4)


def encode_targets(
    boxes: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    input_size: int,
    stride: int,
    max_objects: int = MAX_OBJECTS,
) -> dict[str, np.ndarray]:
    """Encode (N, 4) xyxy input-pixel boxes into dense training targets."""
    grid = input_size // stride
    heatmap = np.zeros((num_classes, grid, grid), dtype=np.float32)
    index = np.zeros(max_objects, dtype=np.int64)
    mask = np.zeros(max_objects, dtype=np.float32)
    size = np.zeros((max_objects, 2), dtype=np.float32)
    offset = np.zeros((max_objects, 2), dtype=np.float32)

    ys, xs = np.mgrid[0:grid, 0:grid].astype(np.float32)
    # Largest first so that when more than max_objects exist, the easy ones survive.
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    for slot, i in enumerate(order[:max_objects]):
        x0, y0, x1, y1 = boxes[i] / stride
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ix = int(np.clip(np.floor(cx), 0, grid - 1))
        iy = int(np.clip(np.floor(cy), 0, grid - 1))
        sx, sy = gaussian_sigmas(w, h)
        g = np.exp(-((xs - ix) ** 2) / (2 * sx**2) - ((ys - iy) ** 2) / (2 * sy**2))
        c = int(labels[i])
        heatmap[c] = np.maximum(heatmap[c], g)
        heatmap[c, iy, ix] = 1.0
        index[slot] = iy * grid + ix
        mask[slot] = 1.0
        size[slot] = np.log(np.maximum([w, h], 1e-3))
        offset[slot] = (cx - ix, cy - iy)
    return {"heatmap": heatmap, "index": index, "mask": mask, "size": size, "offset": offset}


@torch.no_grad()
def decode(
    heat_logits: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    stride: int,
    top_k: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw head outputs to (B, k, 4) xyxy boxes, (B, k) scores, (B, k) class ids."""
    b, c, h, w = heat_logits.shape
    heat = heat_logits.sigmoid()
    peaks = heat == F.max_pool2d(heat, 3, stride=1, padding=1)
    scores, flat = (heat * peaks).view(b, -1).topk(min(top_k, c * h * w))
    classes = flat // (h * w)
    cell = flat % (h * w)
    cy, cx = (cell // w).float(), (cell % w).float()

    def gather(t: torch.Tensor) -> torch.Tensor:
        return t.view(b, 2, h * w).gather(2, cell.unsqueeze(1).expand(b, 2, -1))

    off, wh = gather(offset), gather(size).exp()
    cx, cy = cx + off[:, 0], cy + off[:, 1]
    boxes = torch.stack(
        [cx - wh[:, 0] / 2, cy - wh[:, 1] / 2, cx + wh[:, 0] / 2, cy + wh[:, 1] / 2], dim=-1
    )
    return boxes * stride, scores, classes
