"""The board's JPEG decoder on the PC (tinyseek.deploy.tjpgd) and reference.board_input.

The strongest check is against the board itself: run_003/004/006 of the backpack stage
hold the int8 input tensors the board computed from its own decode of eight camera JPEGs
(arduino-esp32 3.3.12, ROM TJpgDec). Host TJpgDec plus the Pillow-exact preprocessing must
reproduce them byte for byte. That test skips when those results are not on this machine.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tinyseek.deploy import tjpgd
from tinyseek.deploy.reference import (
    BOARD_MODES,
    BoardMode,
    board_input,
    board_rgb,
    box_average,
    decode_output,
    model_input,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "runs" / "deploy" / "stage_backpack"
BOARD_RUNS = [STAGE / "results" / r for r in ("run_003", "run_004", "run_006")]

pytestmark = pytest.mark.skipif(not tjpgd.available(), reason="no C compiler for the host TJpgDec")


@pytest.fixture(autouse=True)
def _cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("TINYSEEK_BUILD_CACHE", str(tmp_path_factory.getbasetemp() / "build"))


def jpeg_bytes(w: int, h: int, subsampling: int, seed: int = 0, quality: int = 85) -> bytes:
    """A smooth, moderately coloured random image (camera-like) as baseline JPEG bytes."""
    rng = np.random.default_rng(seed)
    small = rng.integers(64, 192, (h // 32 + 1, w // 32 + 1, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, subsampling=subsampling)
    return buf.getvalue()


@pytest.mark.parametrize("subsampling", [0, 1, 2])  # 4:4:4, 4:2:2 (the OV2640's), 4:2:0
def test_full_scale_decode_is_close_to_pillow(subsampling):
    data = jpeg_bytes(800, 600, subsampling)
    ours = tjpgd.decode(data, 0).astype(int)
    pil = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), int)
    assert ours.shape == pil.shape == (600, 800, 3)
    # Different IDCT and chroma upsampling (TJpgDec replicates chroma), the same picture:
    # about 1.2-1.4 uint8 steps here, 1.0 on the camera photos.
    assert np.abs(ours - pil).mean() < 2.0


@pytest.mark.parametrize("subsampling", [0, 1, 2])
@pytest.mark.parametrize("scale", [1, 2])
def test_scaled_decode_is_a_truncating_box_average(subsampling, scale):
    """TJpgDec R0.01b builds the full-resolution RGB MCU, then averages 2^s x 2^s squares
    with a right shift: so scale s is exactly box_average of the scale-0 output."""
    data = jpeg_bytes(800, 600, subsampling, seed=scale)
    full = tjpgd.decode(data, 0)
    assert np.array_equal(tjpgd.decode(data, scale), box_average(full, 1 << scale))


def test_eighth_scale_uses_dc_only():
    data = jpeg_bytes(800, 600, 1)
    eighth = tjpgd.decode(data, 3)
    assert eighth.shape == (75, 100, 3)
    # DC only, with one chroma value per MCU: near the block average of the full decode but
    # not equal (about 3 uint8 steps here for 4:2:2), so decode_scale=3 is the coarse option.
    assert np.abs(eighth.astype(int) - box_average(tjpgd.decode(data, 0), 8).astype(int)).mean() < 8


def test_rejects_what_the_board_rejects(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), 50).save(buf, "JPEG", progressive=True)
    with pytest.raises(tjpgd.TjpgdError):
        tjpgd.decode(buf.getvalue())
    buf = io.BytesIO()
    Image.new("L", (64, 64), 50).save(buf, "JPEG")
    with pytest.raises(tjpgd.TjpgdError):
        tjpgd.decode(buf.getvalue())
    with pytest.raises(ValueError):
        tjpgd.decode(jpeg_bytes(16, 16, 0), 4)


def _board_cases():
    cases = []
    for run in BOARD_RUNS:
        for model, size in (("px192", 192), ("px128", 128)):
            folder = run / "parity" / model
            if folder.exists():
                cases += [(run.name, model, size, p.name.split("_")[0]) for p in sorted(folder.glob("*_jpg_in.bin"))]
    return cases


@pytest.mark.skipif(not _board_cases(), reason="board parity results (runs/deploy/stage_backpack) not present")
def test_reproduces_the_board_jpeg_path_bit_exactly():
    """Board tensors from its own ROM TJpgDec decode == host TJpgDec + the PC reference."""
    for run, model, size, case in _board_cases():
        board = np.fromfile(STAGE / "results" / run / "parity" / model / f"{case}_jpg_in.bin", np.int8)
        q, _ = board_input(STAGE / "sdcard" / "tinyseek_eval" / "parity" / f"{case}.jpg", size, -6, "full")
        assert np.array_equal(q.reshape(-1), board), (run, model, case)


def test_board_mode_names_round_trip():
    for name in BOARD_MODES:
        assert BoardMode.parse(name).name == name
    m = BoardMode.parse("quarter_bilinear_pillowdecode")
    assert (m.decode_scale, m.resample, m.decoder) == (2, "bilinear", "pillow")
    for bad in ("double", "half_nearest", "full_bilinear_x"):
        with pytest.raises(ValueError):
            BoardMode.parse(bad)


@pytest.mark.parametrize("mode", BOARD_MODES)
def test_board_input_letterbox_is_in_frame_pixels(mode):
    """Whatever the decode scale, the letterbox maps the model canvas back onto 800x600."""
    data = jpeg_bytes(800, 600, 1, seed=3)
    q, lb = board_input(data, 192, -6, mode)
    assert q.shape == (192, 192, 3) and q.dtype == np.int8
    corners = lb.inverse(np.array([[lb.pad_x, lb.pad_y, 192 - lb.pad_x, 192 - lb.pad_y]], np.float32))[0]
    assert np.allclose(corners, [0, 0, 800, 600], atol=1e-3)
    # A planted peak decodes to the same frame box in every mode.
    out = np.full((24, 24, 5), -128, np.int8)
    out[12, 10] = [100, 32, 32, 0, 0]  # log_w = log_h = 2 (exponent -4): 7.39 cells
    (det,) = decode_output(out, -4, lb)
    assert abs(det.x + det.w / 2 - 10 * 8 / 0.24) < 1e-3 and abs(det.y + det.h / 2 - (12 * 8 - 24) / 0.24) < 1e-3


def test_full_mode_with_pillow_decoder_is_the_pc_path():
    data = jpeg_bytes(800, 600, 1, seed=4)
    q, lb = board_input(data, 192, -6, "full_pillowdecode")
    ref, ref_lb = model_input(Image.open(io.BytesIO(data)), 192, -6)
    assert np.array_equal(q, ref) and lb == ref_lb


def test_board_rgb_shapes():
    data = jpeg_bytes(800, 600, 1, seed=5)
    assert [board_rgb(data, s).shape for s in range(4)] == [(600, 800, 3), (300, 400, 3), (150, 200, 3), (75, 100, 3)]
    assert board_rgb(data, 1, "pillow").shape == (300, 400, 3)
