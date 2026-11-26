"""The SD-card layout shared by the staging tool, the firmware and the comparison.

Card (copy ``<stage>/sdcard/tinyseek_eval`` to the card root)::

    /tinyseek_eval/bench.txt            key=value benchmark settings
    /tinyseek_eval/bench.jpg            one 800x600 camera frame (preprocess / JPEG timing)
    /tinyseek_eval/models/<m>.espdl     one model per file; <m> is a short name (px192)
    /tinyseek_eval/models/<m>.txt       key=value model metadata (decode settings, source)
    /tinyseek_eval/parity/cases.txt     one case id per line
    /tinyseek_eval/parity/<id>.jpg      the camera JPEG
    /tinyseek_eval/parity/<id>.rgb      the same frame decoded on the PC (RGB888, row-major)
    /tinyseek_eval/results/run_NNN/     written by the board, one folder per boot

Host only, next to ``sdcard/``::

    stage.json                          what was staged, from which runs, with hashes
    expected/<m>/<id>_in.bin            PC int8 input tensor (HWC) for <id>.rgb
    expected/<m>/<id>_out.bin           esp-ppq simulated int8 output (NHWC)
    expected/<m>/<id>_float.json        float-model detections (for the quantization drop)

The firmware writes, per run::

    bench.csv, memory.csv               timings and heap snapshots (see the README)
    parity/<m>/<id>_rgb_in.bin          its input tensor from <id>.rgb (must equal expected)
    parity/<m>/<id>_rgb_out.bin         its raw output for that input
    parity/<m>/<id>_jpg_in.bin          its input tensor from decoding <id>.jpg itself
    parity/<m>/<id>_jpg_out.bin
    parity/<m>/dets.csv                 model,case,path,rank,x,y,w,h,score,cls
"""

from __future__ import annotations

from pathlib import Path

CARD_ROOT = "tinyseek_eval"
DETS_HEADER = ["model", "case", "path", "rank", "x", "y", "w", "h", "score", "cls"]


def write_kv(path: Path, values: dict) -> None:
    """key=value lines: the firmware parses these without a JSON library."""
    lines = [f"{k}={v}" for k, v in values.items()]
    Path(path).write_text("\n".join(lines) + "\n")


def read_kv(path: Path) -> dict[str, str]:
    """The firmware's rules (tinyseek_core kv.cpp): BOM dropped, lines and keys/values trimmed,
    '#' lines skipped, a repeated key keeps its last value."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out
