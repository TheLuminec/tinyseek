"""PC reference for what the board computes, in the board's tensor formats.

The board feeds ESP-DL an int8 NHWC tensor and gets an int8 NHWC tensor back. These
helpers produce and consume exactly those, reusing the PC evaluation code
(``letterbox``, ``to_tensor``, ``decode``, ``Letterbox.inverse``) so that "the PC result"
means what ``tinyseek.eval.evaluate.predict`` would report, with only the esp-ppq
quantization step added (power-of-two scale, ROUND_HALF_UP, int8 saturation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tinyseek.config import input_wh
from tinyseek.data.dataset import to_tensor
from tinyseek.data.transforms import Letterbox, letterbox, letterbox_geometry
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
    """esp-ppq's power-of-two quantization for the ESP32-S3: floor(x / 2^exp + 0.5), saturated.

    The same rule as esp-dl 3.3.11's quantize<int8_t/int16_t> (round_half_up, then clip to
    the type's range), so it holds for int16 tensors (bits=16) as well."""
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    t = np.asarray(x, np.float32) / np.float32(2.0**exponent)
    return np.clip(np.floor(t + np.float32(0.5)), lo, hi).astype(np.int8 if bits == 8 else np.int16)


def dequantize(q: np.ndarray, exponent: int) -> np.ndarray:
    return q.astype(np.float32) * np.float32(2.0**exponent)


Size = int | list[int] | tuple[int, int]  # model input: an int (square) or (width, height)


def model_input(image: Image.Image, size: Size, exponent: int, bits: int = 8,
                pad_value: int = 114) -> tuple[np.ndarray, Letterbox]:
    """(height, width, 3) int8 (bits=16: int16) HWC input for one RGB frame, plus its letterbox.

    ``size`` is the model input, an int or (width, height). ``pad_value`` is the letterbox
    fill as a uint8 pixel (the sidecar's pad_value; the PC evaluation always uses 114)."""
    image = image.convert("RGB")
    canvas, lb = letterbox(image, size)
    if pad_value != 114:
        in_w, in_h = input_wh(size)
        _, nw, nh, _, _ = letterbox_geometry(image.width, image.height, size)
        arr = np.full((in_h, in_w, 3), pad_value, np.uint8)
        box = (slice(lb.pad_y, lb.pad_y + nh), slice(lb.pad_x, lb.pad_x + nw))
        arr[box] = np.asarray(canvas)[box]
        canvas = Image.fromarray(arr)
    chw = to_tensor(canvas).numpy()
    return quantize(chw.transpose(1, 2, 0), exponent, bits), lb


def io_bits(dtype: str) -> int:
    """8 or 16 from an esp-ppq/ESP-DL dtype name ("INT8", "INT16")."""
    d = dtype.upper()
    if d not in ("INT8", "INT16"):
        raise ValueError(f"model I/O must be INT8 or INT16, got {dtype!r}")
    return 16 if d == "INT16" else 8


# --- What the detector firmware feeds the model, per speed setting --------------------------

RESAMPLES = ("pillow", "bilinear")
DECODERS = ("tjpgd", "pillow")
SCALE_NAMES = {"full": 0, "half": 1, "quarter": 2, "eighth": 3}


@dataclass(frozen=True)
class BoardMode:
    """One preprocessing setting of firmware/detector (config.txt ``decode_scale``, ``resize``).

    decode_scale: the camera JPEG is decoded at 1/2**decode_scale (TJpgDec's built-in
        descaling: each output pixel is the truncated mean of a 2**s x 2**s block).
    resample: "pillow" is the exact PC letterbox (Pillow BILINEAR), "bilinear" the fast
        2-tap interpolation (tinyseek_core Resample::kBilinear).
    decoder: "tjpgd" is the board's own decoder, bit-exact (needs a C compiler, see
        tinyseek.deploy.tjpgd); "pillow" decodes with Pillow (libjpeg) and applies the
        same truncating box average, an approximation (about 1 uint8 step mean absolute
        difference at full scale on the camera photos).
    """

    decode_scale: int = 0
    resample: str = "pillow"
    decoder: str = "tjpgd"

    def __post_init__(self) -> None:
        if self.decode_scale not in (0, 1, 2, 3):
            raise ValueError(f"decode_scale must be 0..3, got {self.decode_scale}")
        if self.resample not in RESAMPLES:
            raise ValueError(f"resample must be one of {RESAMPLES}, got {self.resample!r}")
        if self.decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {self.decoder!r}")

    @classmethod
    def parse(cls, mode: BoardMode | str) -> BoardMode:
        """"full", "half", "quarter", "eighth", optionally with "_bilinear" (e.g.
        "half_bilinear") and/or "_pillowdecode"; "full" is what perf_eval and the detector's
        defaults run."""
        if isinstance(mode, BoardMode):
            return mode
        parts = mode.lower().split("_")
        if parts[0] not in SCALE_NAMES:
            raise ValueError(f"unknown board mode {mode!r}; start with one of {sorted(SCALE_NAMES)}")
        kw: dict = {"decode_scale": SCALE_NAMES[parts[0]]}
        for p in parts[1:]:
            if p in RESAMPLES:
                kw["resample"] = p
            elif p == "pillowdecode":
                kw["decoder"] = "pillow"
            else:
                raise ValueError(f"unknown part {p!r} in board mode {mode!r}")
        return cls(**kw)

    @property
    def name(self) -> str:
        scale = {v: k for k, v in SCALE_NAMES.items()}[self.decode_scale]
        return scale + ("_bilinear" if self.resample == "bilinear" else "") + (
            "_pillowdecode" if self.decoder == "pillow" else "")


BOARD_MODES = tuple(f"{s}{r}" for s in ("full", "half", "quarter") for r in ("", "_bilinear"))


def box_average(rgb: np.ndarray, factor: int) -> np.ndarray:
    """TJpgDec's descaling: truncated mean of each factor x factor block; partial blocks at
    the right and bottom edges are dropped (the output is floor(W/f) x floor(H/f))."""
    if factor == 1:
        return rgb
    h, w = rgb.shape[0] // factor, rgb.shape[1] // factor
    blocks = rgb[: h * factor, : w * factor].astype(np.uint32).reshape(h, factor, w, factor, 3)
    return (blocks.sum(axis=(1, 3)) // (factor * factor)).astype(np.uint8)


def board_rgb(jpeg: Path | str | bytes, decode_scale: int = 0, decoder: str = "tjpgd") -> np.ndarray:
    """The decoded frame the board preprocesses: (H >> s, W >> s, 3) uint8 RGB."""
    if decoder == "tjpgd":
        from tinyseek.deploy import tjpgd

        return tjpgd.decode(jpeg, decode_scale)
    import io

    src = io.BytesIO(jpeg) if isinstance(jpeg, bytes) else jpeg
    rgb = np.asarray(Image.open(src).convert("RGB"), np.uint8)
    return box_average(rgb, 1 << decode_scale)


def bilinear2_axis(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per output index: first source index and the next one's weight out of 256
    (tinyseek_core bilinear2_axis)."""
    i = np.arange(out_size, dtype=np.int64)
    num = (2 * i + 1) * in_size - out_size
    p = np.where(num > 0, (num * 256) // (2 * out_size), 0)
    i0, f = p >> 8, p & 255
    past = i0 >= in_size - 1
    return np.where(past, in_size - 1, i0), np.where(past, 0, f)


def bilinear_resize(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """The firmware's fast resample (Resample::kBilinear), bit-exact: 8-bit fixed-point
    weights, uint16 horizontal pass, one rounding at the end."""
    h, w = rgb.shape[:2]
    x0, fx = bilinear2_axis(w, out_w)
    y0, fy = bilinear2_axis(h, out_h)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    src = rgb.astype(np.int64)
    fx3 = fx[None, :, None]
    horiz = src[:, x0] * (256 - fx3) + src[:, x1] * fx3  # (h, out_w, 3), at most 65280
    fy3 = fy[:, None, None]
    v = horiz[y0] * (256 - fy3) + horiz[y1] * fy3
    return ((v + 32768) >> 16).astype(np.uint8)


def preprocess_rgb(rgb: np.ndarray, size: Size, exponent: int, resample: str = "pillow",
                   bits: int = 8, pad_value: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Decoded frame -> (height, width, 3) int8 (or int16) HWC input and its letterbox (rgb's pixels)."""
    if resample == "pillow":
        return model_input(Image.fromarray(rgb), size, exponent, bits, pad_value)
    if resample != "bilinear":
        raise ValueError(f"resample must be one of {RESAMPLES}, got {resample!r}")
    h, w = rgb.shape[:2]
    in_w, in_h = input_wh(size)
    scale, nw, nh, pad_x, pad_y = letterbox_geometry(w, h, size)
    canvas = np.full((in_h, in_w, 3), pad_value, np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = bilinear_resize(rgb, nw, nh)
    chw = to_tensor(Image.fromarray(canvas)).numpy()
    return quantize(chw.transpose(1, 2, 0), exponent, bits), Letterbox(scale=scale, pad_x=pad_x, pad_y=pad_y)


def board_input(jpeg: Path | str | bytes, size: Size, exponent: int,
                mode: BoardMode | str = "full", bits: int = 8) -> tuple[np.ndarray, Letterbox]:
    """The int8 (bits=16: int16, for w16a16 models) HWC tensor the detector feeds ESP-DL.

    ``size`` is the model input, an int or (width, height); the tensor is (height, width, 3).
    Returns the tensor and the letterbox in the JPEG's own (full-frame) pixels, so
    ``decode_output(q_out, out_exp, lb)`` gives boxes in frame coordinates, exactly as the
    board reports them. ``mode`` is a BoardMode or its name ("full", "half_bilinear", ...);
    "full" is the default firmware setting and perf_eval's path.
    """
    m = BoardMode.parse(mode)
    rgb = board_rgb(jpeg, m.decode_scale, m.decoder)
    q, lb = preprocess_rgb(rgb, size, exponent, m.resample, bits)
    return q, frame_letterbox(lb, m.decode_scale)


def frame_letterbox(lb: Letterbox, decode_scale: int) -> Letterbox:
    """A decoded-frame letterbox in full-frame pixels (tinyseek_core frame_letterbox)."""
    return Letterbox(scale=lb.scale / 2**decode_scale, pad_x=lb.pad_x, pad_y=lb.pad_y)


# log_w/log_h are clamped to this before exp, as tinyseek_core's decode does (kMaxLogWH):
# no real output comes near it; it only keeps degenerate outputs finite.
MAX_LOG_WH = 16.0


def nhwc_to_heads(out: np.ndarray, num_classes: int) -> dict[str, torch.Tensor]:
    """(grid_h, grid_w, C) float output in NHWC -> the NCHW head dict ``decode`` takes."""
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
    """Board-format int8 or int16 output (grid_h, grid_w, C) -> detections in original pixels,
    as ``predict`` does. The grid need not be square; the letterbox has one scale for both axes."""
    heads = nhwc_to_heads(dequantize(q_out, exponent), num_classes)
    size = heads["size"].clamp(max=MAX_LOG_WH)
    boxes, scores, classes = decode(heads["heatmap"], size, heads["offset"], stride, max_detections)
    keep = scores[0] >= score_threshold
    xyxy = lb.inverse(boxes[0][keep].numpy())
    dets = [
        Detection(float(x0), float(y0), float(x1 - x0), float(y1 - y0), float(s), int(c))
        for (x0, y0, x1, y1), s, c in zip(xyxy, scores[0][keep].numpy(), classes[0][keep].numpy())
    ]
    return sorted(dets, key=lambda d: -d.score)
