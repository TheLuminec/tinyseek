"""Letterboxing and augmentation for low-resolution detector inputs.

Boxes are float arrays of shape (N, 4) in ``x0, y0, x1, y1`` pixel coordinates. A ``size``
is the model input as ``tinyseek.config.input_wh`` takes it: an int (square) or
``(width, height)``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from tinyseek.config import input_wh

PAD_VALUE = (114, 114, 114)


@dataclass(frozen=True)
class Letterbox:
    """Maps original-image coordinates to the model input: ``x' = x * scale + pad_x``.

    One scale for both axes, whatever the input's shape."""

    scale: float
    pad_x: float
    pad_y: float

    def forward(self, boxes: np.ndarray) -> np.ndarray:
        out = boxes.astype(np.float32).copy()
        out[:, [0, 2]] = out[:, [0, 2]] * self.scale + self.pad_x
        out[:, [1, 3]] = out[:, [1, 3]] * self.scale + self.pad_y
        return out

    def inverse(self, boxes: np.ndarray) -> np.ndarray:
        out = boxes.astype(np.float32).copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_x) / self.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_y) / self.scale
        return out


def letterbox_geometry(
    w: int, h: int, size: int | tuple[int, int], zoom: float = 1.0,
    shift: tuple[float, float] = (0.5, 0.5),
) -> tuple[float, int, int, int, int]:
    """(scale, resized_w, resized_h, pad_x, pad_y) of a ``w`` x ``h`` image in the input.

    The image is scaled by ``min(W / w, H / h) * zoom`` (it fits the input at zoom 1, with
    no padding when the aspect ratios match) and placed by ``shift``. For a square input
    the scale is ``size * zoom / max(w, h)``, bit for bit: the smaller of two quotients
    with one numerator is the one with the larger denominator.
    tinyseek_core's compute_letterbox is the same arithmetic at zoom 1, centred.
    """
    in_w, in_h = input_wh(size)
    scale = min(in_w * zoom / w, in_h * zoom / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return scale, nw, nh, round((in_w - nw) * shift[0]), round((in_h - nh) * shift[1])


def letterbox(
    image: Image.Image, size: int | tuple[int, int], zoom: float = 1.0,
    shift: tuple[float, float] = (0.5, 0.5),
) -> tuple[Image.Image, Letterbox]:
    """Resize to fit the input (times ``zoom``) and paste onto a width x height canvas.

    ``shift`` in [0, 1] picks where the resized image sits (0.5 = centered). With
    ``zoom > 1`` the image overflows and the canvas becomes a random crop.
    """
    scale, nw, nh, pad_x, pad_y = letterbox_geometry(*image.size, size, zoom, shift)
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", input_wh(size), PAD_VALUE)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, Letterbox(scale=scale, pad_x=pad_x, pad_y=pad_y)


def clip_boxes(
    boxes: np.ndarray, size: int | tuple[int, int], min_visible: float = 0.4, min_side: float = 2.0
) -> np.ndarray:
    """Clip boxes to the canvas, dropping ones mostly cut off or too small to learn."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    in_w, in_h = input_wh(size)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    clipped = np.clip(boxes, 0, np.asarray([in_w, in_h, in_w, in_h], dtype=boxes.dtype))
    w = clipped[:, 2] - clipped[:, 0]
    h = clipped[:, 3] - clipped[:, 1]
    keep = (w >= min_side) & (h >= min_side) & (w * h >= min_visible * np.maximum(area, 1e-6))
    return clipped[keep]


def color_jitter(image: Image.Image, strength: float, rng: np.random.Generator) -> Image.Image:
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        image = enhancer(image).enhance(1.0 + rng.uniform(-strength, strength))
    return image


def sensor_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Approximate a cheap OV2640-class sensor: softness, read noise, heavy JPEG."""
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
    if rng.random() < 0.7:
        arr = np.asarray(image, dtype=np.float32)
        arr += rng.normal(0.0, rng.uniform(2.0, 10.0), arr.shape)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.7:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(rng.integers(10, 50)))
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
    return image


def augment(
    image: Image.Image,
    boxes: np.ndarray,
    size: int | tuple[int, int],
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[Image.Image, np.ndarray]:
    """Random zoom/placement letterbox, horizontal flip, color jitter, and sensor noise."""
    lo, hi = cfg.get("scale_jitter", [1.0, 1.0])
    canvas, lb = letterbox(image, size, zoom=rng.uniform(lo, hi), shift=tuple(rng.uniform(0, 1, 2)))
    boxes = clip_boxes(lb.forward(boxes), size)
    if rng.random() < cfg.get("hflip", 0.0):
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = boxes.copy()
        boxes[:, [0, 2]] = input_wh(size)[0] - boxes[:, [2, 0]]
    if cfg.get("color_jitter", 0.0) > 0:
        canvas = color_jitter(canvas, cfg["color_jitter"], rng)
    if cfg.get("sensor_noise", False):
        canvas = sensor_noise(canvas, rng)
    return canvas, boxes
