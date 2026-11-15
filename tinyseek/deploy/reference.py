"""PC reference for what the board computes, in the board's tensor formats.

The board feeds ESP-DL an int8 NHWC tensor and gets an int8 NHWC tensor back. These
helpers produce and consume exactly those, reusing the PC evaluation code
(``letterbox``, ``to_tensor``, ``decode``, ``Letterbox.inverse``) so that "the PC result"
means what ``tinyseek.eval.evaluate.predict`` would report, with only the esp-ppq
quantization step added (power-of-two scale, ROUND_HALF_UP, int8 saturation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from tinyseek.data.dataset import to_tensor
from tinyseek.data.transforms import Letterbox, letterbox
from tinyseek.train.targets import decode


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    cls: int = 0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def quantize(x: np.ndarray, exponent: int, bits: int = 8) -> np.ndarray:
    """esp-ppq's power-of-two quantization for the ESP32-S3: floor(x / 2^exp + 0.5), saturated."""
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    t = np.asarray(x, np.float32) / np.float32(2.0**exponent)
    return np.clip(np.floor(t + np.float32(0.5)), lo, hi).astype(np.int8 if bits == 8 else np.int16)


def dequantize(q: np.ndarray, exponent: int) -> np.ndarray:
    return q.astype(np.float32) * np.float32(2.0**exponent)


def model_input(image: Image.Image, size: int, exponent: int) -> tuple[np.ndarray, Letterbox]:
    """(size, size, 3) int8 HWC input for one RGB frame, plus its letterbox transform."""
    canvas, lb = letterbox(image.convert("RGB"), size)
    chw = to_tensor(canvas).numpy()
    return quantize(chw.transpose(1, 2, 0), exponent), lb


def nhwc_to_heads(out: np.ndarray, num_classes: int) -> dict[str, torch.Tensor]:
    """(G, G, C) float output in NHWC -> the NCHW head dict ``decode`` takes."""
    t = torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1)))[None].float()
    c = num_classes
    return {"heatmap": t[:, :c], "size": t[:, c : c + 2], "offset": t[:, c + 2 : c + 4]}


def decode_output(
    q_out: np.ndarray,
    exponent: int,
    lb: Letterbox,
    num_classes: int = 1,
    stride: int = 8,
    score_threshold: float = 0.05,
    max_detections: int = 20,
) -> list[Detection]:
    """Board-format int8 output (G, G, C) -> detections in original pixels, as ``predict`` does."""
    heads = nhwc_to_heads(dequantize(q_out, exponent), num_classes)
    boxes, scores, classes = decode(heads["heatmap"], heads["size"], heads["offset"], stride, max_detections)
    keep = scores[0] >= score_threshold
    xyxy = lb.inverse(boxes[0][keep].numpy())
    dets = [
        Detection(float(x0), float(y0), float(x1 - x0), float(y1 - y0), float(s), int(c))
        for (x0, y0, x1, y1), s, c in zip(xyxy, scores[0][keep].numpy(), classes[0][keep].numpy())
    ]
    return sorted(dets, key=lambda d: -d.score)
