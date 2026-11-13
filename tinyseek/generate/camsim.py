"""Simulate the ESP32 camera on clean images, moving boxes with the pixels.

An analytic alternative to restyling with the domain LoRA: it adds the camera's look but
cannot add or remove objects, and the boxes are transformed exactly, so the teacher's labels
on the clean image stay valid.

  1. barrel distortion (wide-angle lens): output pixel at normalized radius r samples the
     source at r * s * (1 + k r^2); s keeps the corners inside the source. Boxes are mapped
     by inverting that radius function on points along their edges.
  2. tone: brightness and contrast pulled to targets sampled from the real photos
     (auto-exposure keeps their mean luma ~59-67, contrast std ~34-59)
  3. saturation scaled toward a target sampled from the real photos (~12-33 %)
  4. vignetting, then the training augmentation's sensor noise (blur, noise, JPEG)

    python -m tinyseek.generate.camsim --src data/synthetic/backpack/plain1k \
        --labels data/labels/owlv2/synth_plain1k_backpack_t0.3.json \
        --out data/synthetic/backpack/plain1k_camsim --out-labels data/labels/owlv2/synth_plain1k_camsim_backpack_t0.3.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tinyseek.data.transforms import sensor_noise


@dataclass(frozen=True)
class CamSim:
    k: float = 0.35                     # barrel strength; 0 = no distortion
    # Targets before vignetting/noise; tuned so outputs land on the real photos' ranges
    # (measured: luma 59-67, saturation 12-33 %; contrast gain adds some saturation back).
    luma: tuple[float, float] = (62.0, 72.0)
    contrast: tuple[float, float] = (34.0, 59.0)
    saturation: tuple[float, float] = (0.09, 0.24)
    tone: bool = True                   # match brightness, contrast and saturation to the targets above
    vignette: float = 0.35              # darkening at the corners, 0..1
    noise: bool = True


PRESETS = {
    "mild": CamSim(k=0.2, vignette=0.2),
    "camera": CamSim(),
    "strong": CamSim(k=0.5, vignette=0.45),
    # One part of "camera" at a time, to measure what each does to a detector.
    "only_distort": CamSim(tone=False, vignette=0.0, noise=False),
    "only_tone": CamSim(k=0.0, vignette=0.0, noise=False),
    "only_vignette": CamSim(k=0.0, tone=False, noise=False),
    "only_noise": CamSim(k=0.0, tone=False, vignette=0.0),
}


def _radius_scale(k: float) -> float:
    """s so that the output corner (r = sqrt 2) samples the source corner."""
    r = np.sqrt(2.0)
    return 1.0 / (1.0 + k * r * r)


def distort(image: Image.Image, k: float) -> Image.Image:
    if k <= 0:
        return image
    w, h = image.size
    s = _radius_scale(k)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    factor = s * (1 + k * (xs**2 + ys**2))
    grid = torch.stack([xs * factor, ys * factor], dim=-1).unsqueeze(0)
    src = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return Image.fromarray(out[0].permute(1, 2, 0).clamp(0, 255).byte().numpy())


def _source_to_output(points: np.ndarray, k: float, size: tuple[int, int]) -> np.ndarray:
    """Map source pixel coords to output pixel coords (inverse of the sampling map)."""
    w, h = size
    u = points[:, 0] / (w - 1) * 2 - 1
    v = points[:, 1] / (h - 1) * 2 - 1
    r_src = np.hypot(u, v)
    s = _radius_scale(k)
    # Solve s * r (1 + k r^2) = r_src for r (monotonic in r >= 0) by Newton's method.
    r = r_src / s
    for _ in range(20):
        f = s * r * (1 + k * r * r) - r_src
        r = r - f / (s * (1 + 3 * k * r * r))
    ratio = np.where(r_src > 1e-9, r / np.maximum(r_src, 1e-9), 1 / s)
    uo, vo = u * ratio, v * ratio
    return np.stack([(uo + 1) / 2 * (w - 1), (vo + 1) / 2 * (h - 1)], axis=1)


def distort_box(bbox: list[float], k: float, size: tuple[int, int], samples: int = 16) -> list[float] | None:
    """xywh box in the source -> tight xywh box around its distorted outline (clipped)."""
    if k <= 0:
        return bbox
    x, y, bw, bh = bbox
    t = np.linspace(0, 1, samples)
    edges = np.concatenate([
        np.stack([x + t * bw, np.full_like(t, y)], 1), np.stack([x + t * bw, np.full_like(t, y + bh)], 1),
        np.stack([np.full_like(t, x), y + t * bh], 1), np.stack([np.full_like(t, x + bw), y + t * bh], 1),
    ])
    out = _source_to_output(edges, k, size)
    w, h = size
    x0, y0 = np.clip(out.min(0), 0, [w, h])
    x1, y1 = np.clip(out.max(0), 0, [w, h])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def tone_and_colour(image: Image.Image, sim: CamSim, rng: np.random.Generator) -> Image.Image:
    """Saturation first, then brightness/contrast, so the brightness target holds at the end
    (desaturating at fixed HSV value raises luminance)."""
    target_mean, target_std = rng.uniform(*sim.luma), rng.uniform(*sim.contrast)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    current_sat = max(float(hsv[..., 1].mean()) / 255, 1e-3)
    hsv[..., 1] = np.clip(hsv[..., 1] * (rng.uniform(*sim.saturation) / current_sat), 0, 255)
    a = np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"), dtype=np.float32)

    def luma(x: np.ndarray) -> np.ndarray:
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    y = luma(a)
    gain = min(target_std / max(float(y.std()), 1e-3), 3.0)
    a = np.clip((a - float(y.mean())) * gain + target_mean, 0, 255)
    # Clipping at 0/255 shifts the mean; re-centre on the brightness target.
    for _ in range(3):
        a = np.clip(a + (target_mean - float(luma(a).mean())), 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def vignette(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return image
    w, h = image.size
    ys, xs = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    mask = 1 - strength * np.clip((xs**2 + ys**2) / 2, 0, 1) ** 1.5
    return Image.fromarray(np.clip(np.asarray(image, dtype=np.float32) * mask[..., None], 0, 255).astype(np.uint8))


def simulate(image: Image.Image, boxes: list[list[float]], sim: CamSim,
             rng: np.random.Generator) -> tuple[Image.Image, list[list[float] | None]]:
    out = distort(image, sim.k)
    new_boxes = [distort_box(b, sim.k, image.size) for b in boxes]
    if sim.tone:
        out = tone_and_colour(out, sim, rng)
    out = vignette(out, sim.vignette)
    if sim.noise:
        out = sensor_noise(out, rng)
    return out, new_boxes


def simulate_set(src: Path, labels: Path | None, out: Path, out_labels: Path | None, sim: CamSim, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    coco = json.loads(labels.read_text()) if labels else None
    by_file: dict[str, list[dict]] = {}
    if coco:
        names = {img["id"]: img["file_name"] for img in coco["images"]}
        for a in coco["annotations"]:
            by_file.setdefault(names[a["image_id"]], []).append(a)
    kept = []
    files = sorted(p.name for p in src.glob("*.jpg"))
    for i, name in enumerate(files):
        rng = np.random.default_rng([seed, i])
        anns = by_file.get(name, [])
        image, boxes = simulate(Image.open(src / name).convert("RGB"), [a["bbox"] for a in anns], sim, rng)
        image.save(out / name, quality=95)
        for a, b in zip(anns, boxes):
            if b is not None:
                kept.append(dict(a, id=len(kept) + 1, bbox=b, area=b[2] * b[3]))
    meta = {"camsim": asdict(sim), "seed": seed, "src": str(src), "labels": str(labels) if labels else None}
    (out / "camsim.json").write_text(json.dumps(meta, indent=1))
    if coco and out_labels:
        out_labels.write_text(json.dumps(dict(coco, annotations=kept, info=dict(coco.get("info", {}), camsim=meta))))
    print(f"{out}: {len(files)} images" + (f", {len(kept)} boxes -> {out_labels}" if coco else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="COCO labels of --src to transform with the images")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="camera")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    simulate_set(args.src, args.labels, args.out, args.out_labels, PRESETS[args.preset], args.seed)


if __name__ == "__main__":
    main()
