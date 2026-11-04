"""Check ESP32 capture sessions for exposure and color cast, and write a contact sheet.

    python -m tinyseek.data.camera_check data/esp32cam/raw [--sheet contact.jpg]

Per image: mean brightness (luma, 0-255) and the green/red and green/blue ratios of the
channel means. A neutral scene (white paper, a gray card) under good light should give
ratios near 1.0; consistently > 1.3 there means a green cast that firmware settings have
not fixed. Very low luma (< 60) means underexposure, which also skews auto white balance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def image_stats(path: Path) -> dict[str, float]:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    r, g, b = (float(a[..., c].mean()) for c in range(3))
    luma = float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())
    return {"luma": luma, "g_r": g / max(r, 1e-6), "g_b": g / max(b, 1e-6)}


def contact_sheet(paths: list[Path], out: Path, cols: int = 6, thumb: tuple[int, int] = (240, 180)) -> None:
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb[0], max(rows, 1) * thumb[1]))
    for i, p in enumerate(paths):
        t = Image.open(p).convert("RGB").resize(thumb)
        ImageDraw.Draw(t).text((4, 4), f"{p.parent.name[-3:]}/{p.stem[-5:]}", fill=(255, 255, 255))
        sheet.paste(t, ((i % cols) * thumb[0], (i // cols) * thumb[1]))
    sheet.save(out, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="folder holding session_* directories")
    parser.add_argument("--sheet", type=Path, help="write a contact sheet JPEG here")
    args = parser.parse_args()

    paths = sorted(args.root.glob("session_*/img_*.jpg"))
    if not paths:
        raise SystemExit(f"no session_*/img_*.jpg under {args.root}")
    by_session: dict[str, list[dict[str, float]]] = {}
    for p in paths:
        by_session.setdefault(p.parent.name, []).append(image_stats(p))

    print(f"{'session':14s} {'images':>6s} {'luma':>6s} {'G/R':>6s} {'G/B':>6s}   (medians)")
    for name, stats in by_session.items():
        med = {k: float(np.median([s[k] for s in stats])) for k in ("luma", "g_r", "g_b")}
        flags = []
        if med["luma"] < 60:
            flags.append("underexposed")
        if med["g_r"] > 1.3 and med["g_b"] > 1.3:
            flags.append("green cast")
        elif med["g_b"] < 0.8:
            flags.append("blue cast")
        print(f"{name:14s} {len(stats):6d} {med['luma']:6.1f} {med['g_r']:6.2f} {med['g_b']:6.2f}   {', '.join(flags)}")
    if args.sheet:
        contact_sheet(paths, args.sheet)
        print(f"contact sheet: {args.sheet}")


if __name__ == "__main__":
    main()
