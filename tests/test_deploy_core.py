"""The firmware's portable C++ (firmware/perf_eval/components/tinyseek_core) against the PC path.

Builds firmware/perf_eval/host_test/core_cli.cpp with g++ and checks that preprocessing is
byte-identical to tinyseek.deploy.reference.model_input (Pillow letterbox + to_tensor +
esp-ppq quantization) and that decode gives the detections tinyseek.train.targets.decode
plus Letterbox.inverse give. Real-image cases skip when the data or runs are absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tinyseek.deploy.reference import Detection, decode_output, model_input, quantize

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware" / "perf_eval"
CAMERA_PHOTO = ROOT / "data" / "esp32cam" / "raw" / "session_013" / "img_00001.jpg"
RUN_192 = ROOT / "runs" / "backpack" / "heldout_qwen21_turbo_far_v1_only_tone_px192" / "seed0"


@pytest.fixture(scope="module")
def cli(tmp_path_factory) -> Path:
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    out = tmp_path_factory.mktemp("core") / "core_cli"
    core = FW / "components" / "tinyseek_core"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", f"-I{core / 'include'}",
         *sorted(str(p) for p in (core / "src").glob("*.cpp")),
         str(FW / "host_test" / "core_cli.cpp"), "-o", str(out)],
        check=True,
    )
    return out


def run_preprocess(cli: Path, image: Image.Image, size: int, exponent: int, tmp: Path) -> tuple[np.ndarray, list[str]]:
    rgb = tmp / "in.rgb"
    np.asarray(image.convert("RGB"), np.uint8).tofile(rgb)
    r = subprocess.run([str(cli), "preprocess", str(rgb), str(image.width), str(image.height), str(size),
                        str(exponent), str(tmp / "out.bin")], capture_output=True, text=True, check=True)
    return np.fromfile(tmp / "out.bin", np.int8).reshape(size, size, 3), r.stdout.split()


def run_decode(cli: Path, q_out: np.ndarray, exponent: int, lb, tmp: Path, thr: float = 0.05, k: int = 20) -> list[Detection]:
    path = tmp / "out.bin"
    q_out.astype(np.int8).tofile(path)
    r = subprocess.run([str(cli), "decode", str(path), str(q_out.shape[0]), str(q_out.shape[2] - 4), "8",
                        str(exponent), repr(lb.scale), str(lb.pad_x), str(lb.pad_y), str(thr), str(k)],
                       capture_output=True, text=True, check=True)
    return [Detection(*map(float, line.split()[:5]), int(line.split()[5])) for line in r.stdout.splitlines()]


def assert_same_dets(ref: list[Detection], got: list[Detection]) -> None:
    assert len(ref) == len(got)
    key = lambda d: (-round(d.score, 5), round(d.x, 2), round(d.y, 2))
    for a, b in zip(sorted(ref, key=key), sorted(got, key=key)):
        assert abs(a.score - b.score) < 1e-6
        assert max(abs(a.x - b.x), abs(a.y - b.y), abs(a.w - b.w), abs(a.h - b.h)) < 1e-3


@pytest.mark.parametrize("wh", [(800, 600), (333, 217), (64, 200), (100, 100)])
@pytest.mark.parametrize("size", [128, 192])
def test_preprocess_matches_pc_on_random_frames(cli, tmp_path, wh, size):
    rng = np.random.default_rng(sum(wh) + size)
    image = Image.fromarray(rng.integers(0, 256, (wh[1], wh[0], 3), dtype=np.uint8))
    got, (scale, pad_x, pad_y, _, _) = run_preprocess(cli, image, size, -6, tmp_path)
    ref, lb = model_input(image, size, -6)
    assert np.array_equal(got, ref)
    assert (float(scale), int(pad_x), int(pad_y)) == (lb.scale, lb.pad_x, lb.pad_y)


@pytest.mark.skipif(not CAMERA_PHOTO.exists(), reason="ESP32 camera photos not present")
@pytest.mark.parametrize("size", [128, 192])
def test_preprocess_matches_pc_on_camera_photo(cli, tmp_path, size):
    image = Image.open(CAMERA_PHOTO).convert("RGB")
    got, _ = run_preprocess(cli, image, size, -6, tmp_path)
    ref, _ = model_input(image, size, -6)
    assert np.array_equal(got, ref)


def test_decode_matches_pc_on_planted_peaks_and_ties(cli, tmp_path):
    _, lb = model_input(Image.new("RGB", (800, 600)), 192, -6)
    rng = np.random.default_rng(0)
    q = rng.integers(-128, -60, (24, 24, 5)).astype(np.int8)  # background: sigmoid < 0.05
    q[..., 1:] = rng.integers(-20, 30, (24, 24, 4))
    q[5, 7, 0], q[12, 3, 0], q[20, 20, 0] = 40, 10, -30
    q[9, 9, 0] = q[9, 10, 0] = 25  # a plateau: both cells are peaks, as with torch's ==
    q[0, 23, 0] = 60  # corner peak: padding must not suppress it
    ref = decode_output(q, -4, lb)
    got = run_decode(cli, q, -4, lb, tmp_path)
    assert len(ref) == 6
    assert_same_dets(ref, got)


def test_decode_respects_max_detections(cli, tmp_path):
    _, lb = model_input(Image.new("RGB", (800, 600)), 128, -6)
    q = np.full((16, 16, 5), -128, np.int8)
    q[::2, ::2, 0] = np.arange(64).reshape(8, 8)  # 64 isolated peaks, all above threshold
    ref = decode_output(q, -4, lb, max_detections=20)
    got = run_decode(cli, q, -4, lb, tmp_path, k=20)
    assert len(ref) == 20
    assert_same_dets(ref, got)


@pytest.mark.skipif(not (CAMERA_PHOTO.exists() and (RUN_192 / "last.pt").exists()), reason="run or photo absent")
def test_decode_matches_pc_on_real_model_output(cli, tmp_path):
    """A real case: the float checkpoint's output on a camera photo, quantized like the int8 output."""
    from tinyseek.config import load_config
    from tinyseek.data.dataset import to_tensor
    from tinyseek.data.transforms import letterbox
    from tinyseek.train.model import ExportWrapper, build_model

    cfg = load_config(RUN_192 / "config.yaml")
    model = build_model(dict(cfg["model"], pretrained=False))
    model.load_state_dict(torch.load(RUN_192 / "last.pt", map_location="cpu")["model"])
    canvas, lb = letterbox(Image.open(CAMERA_PHOTO).convert("RGB"), 192)
    with torch.no_grad():
        out = ExportWrapper(model.eval())(to_tensor(canvas)[None])[0].numpy()
    q = quantize(out.transpose(1, 2, 0), -4)
    ref = decode_output(q, -4, lb)
    assert ref, "expected at least one detection on this photo"
    assert_same_dets(ref, run_decode(cli, q, -4, lb, tmp_path))


def run_jpeginfo(cli: Path, path: Path) -> list[int] | None:
    r = subprocess.run([str(cli), "jpeginfo", str(path)], capture_output=True, text=True, check=False)
    return None if r.returncode else [int(v) for v in r.stdout.split()]


@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_jpeg_info_reads_frame_header(cli, tmp_path, progressive, mode):
    path = tmp_path / "x.jpg"
    exif = Image.Exif()
    exif[0x010F] = "maker"  # an APP1 segment before the frame header
    Image.new(mode, (800, 600), 90).save(path, quality=80, progressive=progressive, exif=exif.tobytes())
    assert run_jpeginfo(cli, path) == [800, 600, 3 if mode == "RGB" else 1, int(progressive)]


def test_jpeg_info_rejects_non_jpeg_and_truncated(cli, tmp_path):
    png = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(png)
    assert run_jpeginfo(cli, png) is None
    jpg = tmp_path / "x.jpg"
    Image.new("RGB", (8, 8)).save(jpg)
    (tmp_path / "cut.jpg").write_bytes(jpg.read_bytes()[:20])
    assert run_jpeginfo(cli, tmp_path / "cut.jpg") is None


@pytest.mark.skipif(not CAMERA_PHOTO.exists(), reason="ESP32 camera photos not present")
def test_jpeg_info_on_camera_photo(cli):
    assert run_jpeginfo(cli, CAMERA_PHOTO) == [*Image.open(CAMERA_PHOTO).size, 3, 0]
