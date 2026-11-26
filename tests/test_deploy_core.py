"""The firmware's portable C++ (firmware/perf_eval/components/tinyseek_core) against the PC path.

Builds firmware/perf_eval/host_test/core_cli.cpp with g++ and checks that preprocessing is
byte-identical to tinyseek.deploy.reference.model_input (Pillow letterbox + to_tensor +
esp-ppq quantization) and that decode gives the detections tinyseek.train.targets.decode
plus Letterbox.inverse give. Real-image cases skip when the data or runs are absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tinyseek.data.transforms import Letterbox
from tinyseek.deploy.reference import (
    Detection,
    bilinear_resize,
    decode_output,
    model_input,
    preprocess_rgb,
    quantize,
)

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


def run_preprocess(cli: Path, image: Image.Image, size: int, exponent: int, tmp: Path,
                   resample: str | None = None, bits: int = 8) -> tuple[np.ndarray, list[str]]:
    rgb = tmp / "in.rgb"
    np.asarray(image.convert("RGB"), np.uint8).tofile(rgb)
    r = subprocess.run([str(cli), "preprocess", str(rgb), str(image.width), str(image.height), str(size),
                        str(exponent), str(tmp / "out.bin"), *([resample] if resample else []),
                        *(["int16"] if bits == 16 else [])],
                       capture_output=True, text=True, check=True)
    dtype = np.int16 if bits == 16 else np.int8
    return np.fromfile(tmp / "out.bin", dtype).reshape(size, size, 3), r.stdout.split()


def run_decode(cli: Path, q_out: np.ndarray, exponent: int, lb, tmp: Path, thr: float = 0.05, k: int = 20) -> list[Detection]:
    """C++ decode of an int8 or int16 output tensor (chosen by q_out's dtype)."""
    path = tmp / "out.bin"
    wide = q_out.dtype == np.int16
    q_out.astype(np.int16 if wide else np.int8).tofile(path)
    r = subprocess.run([str(cli), "decode", str(path), str(q_out.shape[0]), str(q_out.shape[2] - 4), "8",
                        str(exponent), repr(lb.scale), str(lb.pad_x), str(lb.pad_y), str(thr), str(k),
                        *(["int16"] if wide else [])],
                       capture_output=True, text=True, check=True)
    return [Detection(*map(float, line.split()[:5]), int(line.split()[5])) for line in r.stdout.splitlines()]


def assert_same_dets(ref: list[Detection], got: list[Detection]) -> None:
    assert len(ref) == len(got)
    key = lambda d: (-round(d.score, 5), round(d.x, 2), round(d.y, 2))
    for a, b in zip(sorted(ref, key=key), sorted(got, key=key)):
        assert abs(a.score - b.score) < 1e-6
        assert max(abs(a.x - b.x), abs(a.y - b.y), abs(a.w - b.w), abs(a.h - b.h)) < 1e-3


@pytest.mark.parametrize("wh", [(800, 600), (400, 300), (200, 150), (333, 217), (64, 200), (100, 100)])
@pytest.mark.parametrize("size", [128, 192])
def test_preprocess_matches_pc_on_random_frames(cli, tmp_path, wh, size):
    rng = np.random.default_rng(sum(wh) + size)
    image = Image.fromarray(rng.integers(0, 256, (wh[1], wh[0], 3), dtype=np.uint8))
    got, (scale, pad_x, pad_y, _, _) = run_preprocess(cli, image, size, -6, tmp_path)
    ref, lb = model_input(image, size, -6)
    assert np.array_equal(got, ref)
    assert (float(scale), int(pad_x), int(pad_y)) == (lb.scale, lb.pad_x, lb.pad_y)


@pytest.mark.parametrize("wh", [(400, 300), (200, 150), (800, 600), (100, 75), (333, 217), (7, 5), (1, 1)])
@pytest.mark.parametrize("size", [128, 192])
def test_bilinear_resample_matches_pc_reference(cli, tmp_path, wh, size):
    """The fast resample (Resample::kBilinear) against reference.preprocess_rgb, byte for byte."""
    rng = np.random.default_rng(sum(wh) * size)
    rgb = rng.integers(0, 256, (wh[1], wh[0], 3), dtype=np.uint8)
    got, (scale, pad_x, pad_y, _, _) = run_preprocess(cli, Image.fromarray(rgb), size, -6, tmp_path, "bilinear")
    ref, lb = preprocess_rgb(rgb, size, -6, "bilinear")
    assert np.array_equal(got, ref)
    assert (float(scale), int(pad_x), int(pad_y)) == (lb.scale, lb.pad_x, lb.pad_y)


def test_bilinear_resize_reference_properties():
    """Same size is the identity, a flat image stays flat, and 2:1 averages neighbour pairs."""
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)
    assert np.array_equal(bilinear_resize(rgb, 40, 30), rgb)
    flat = np.full((60, 80, 3), 77, np.uint8)
    assert (bilinear_resize(flat, 37, 23) == 77).all()
    # Downscaling by exactly 2 samples halfway between source pixels: rounded pair means.
    half = bilinear_resize(rgb, 20, 15).astype(int)
    a = rgb.astype(int)
    sum4 = a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]
    assert np.array_equal(half, (sum4 + 2) // 4)  # weights 128/256 each way: round half up


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


# --- int16 model I/O (w16a16 exports) ----------------------------------------------------------

@pytest.mark.parametrize("wh", [(800, 600), (400, 300), (200, 150), (333, 217)])
@pytest.mark.parametrize("resample", ["pillow", "bilinear"])
def test_int16_preprocess_matches_reference(cli, tmp_path, wh, resample):
    """int16 input tensor (exponent -13, as the w16a16 export) against the numpy reference."""
    rng = np.random.default_rng(sum(wh) + len(resample))
    rgb = rng.integers(0, 256, (wh[1], wh[0], 3), dtype=np.uint8)
    got, _ = run_preprocess(cli, Image.fromarray(rgb), 192, -13, tmp_path, resample, bits=16)
    ref, _ = preprocess_rgb(rgb, 192, -13, resample, bits=16)
    assert ref.dtype == np.int16 and np.array_equal(got, ref)
    # Same pixels through the int8 path still give the int8 bytes (the paths share the resample).
    got8, _ = run_preprocess(cli, Image.fromarray(rgb), 192, -6, tmp_path, resample)
    assert np.array_equal(got8, preprocess_rgb(rgb, 192, -6, resample)[0])


def test_int16_quantization_rounds_half_up_and_saturates():
    x = np.array([0.5, -0.5, 1.5, -1.5, 1e9, -1e9], np.float32)
    assert quantize(x, 0, 16).tolist() == [1, 0, 2, -1, 32767, -32768]
    assert quantize(np.float32([1 / 8192 * 2.5]), -13, 16).tolist() == [3]  # 2.5 -> 3


def test_int16_decode_matches_pc_including_saturated_ties(cli, tmp_path):
    """With int16 logits, adjacent levels near score 1 share a float32 sigmoid; the PC's
    `heat == max_pool(heat)` then keeps both cells, and so must the board."""
    _, lb = model_input(Image.new("RGB", (800, 600)), 192, -6)
    x = np.arange(28000, 32768, dtype=np.int16)
    s = torch.sigmoid(torch.from_numpy(x.astype(np.float32) * 2.0**-12)).numpy()
    ties = np.where(s[1:] == s[:-1])[0]
    assert len(ties) >= 4
    rng = np.random.default_rng(0)
    q = rng.integers(-32768, -20000, (24, 24, 5)).astype(np.int16)
    q[..., 1:] = rng.integers(-4000, 6000, (24, 24, 4))
    for i, t in enumerate(ties[:4]):  # side-by-side equal-score pairs
        q[2 + 4 * i, 3, 0], q[2 + 4 * i, 4, 0] = x[t], x[t + 1]
    q[20, 20, 0], q[20, 21, 0] = 25000, 26000  # a plain pair: only the larger is a peak
    q[0, 23, 0] = 30000
    ref = decode_output(q, -12, lb)
    assert len(ref) == 4 * 2 + 1 + 1
    assert_same_dets(ref, run_decode(cli, q, -12, lb, tmp_path))
    for seed in range(20):  # random scenes
        r = np.random.default_rng(seed + 1)
        q = r.integers(-32768, -20000, (24, 24, 5)).astype(np.int16)
        q[..., 1:] = r.integers(-4000, 6000, (24, 24, 4))
        idx = r.integers(0, 24, (10, 2))
        q[idx[:, 0], idx[:, 1], 0] = r.integers(-12000, 32767, 10)
        assert_same_dets(decode_output(q, -12, lb), run_decode(cli, q, -12, lb, tmp_path))


INT16_STAGE = ROOT / "runs" / "deploy" / "stage_int16_latency"


def _int16_models() -> list[dict]:
    path = INT16_STAGE / "stage.json"
    if not path.exists():
        return []
    return [m for m in json.loads(path.read_text())["models"] if m.get("input_dtype") == "INT16" and m.get("simulated")]


@pytest.mark.skipif(not _int16_models(), reason="int16 stage (runs/deploy/stage_int16_latency) not present")
def test_int16_parity_with_simulated_w16a16_graph(cli, tmp_path):
    """The staged cases of the w16a16 model: the C++ preprocessing of each frame equals the
    tensor the PC fed esp-ppq's simulated w16a16 graph (whose own float-input run agreed,
    ppq_input_quantization_matches), and the C++ decode of that graph's int16 output equals
    the PC decode."""
    manifest = json.loads((INT16_STAGE / "stage.json").read_text())
    for m in _int16_models():
        assert m["ppq_input_quantization_matches"] is True
        size, grid = m["input_size"], m["input_size"] // m["stride"]
        exp_dir = INT16_STAGE / "expected" / m["name"]
        for case in manifest["cases"][:4]:
            rgb = np.fromfile(INT16_STAGE / "sdcard" / "tinyseek_eval" / "parity" / f"{case['id']}.rgb", np.uint8)
            image = Image.fromarray(rgb.reshape(case["size"][1], case["size"][0], 3))
            got, _ = run_preprocess(cli, image, size, m["input_exponent"], tmp_path, bits=16)
            assert np.array_equal(got.reshape(-1), np.fromfile(exp_dir / f"{case['id']}_in.bin", np.int16))
            out = np.fromfile(exp_dir / f"{case['id']}_out.bin", np.int16).reshape(grid, grid, m["num_classes"] + 4)
            lb = Letterbox(*json.loads((exp_dir / f"{case['id']}_float.json").read_text())["letterbox"])
            ref = decode_output(out, m["output_exponent"], lb)
            assert_same_dets(ref, run_decode(cli, out, m["output_exponent"], lb, tmp_path))


# --- review fixes (2026-11-26) -----------------------------------------------------------------

def run_kv(cli: Path, path: Path) -> dict[str, str]:
    out = subprocess.run([str(cli), "kv", str(path)], capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


def test_kv_trims_keys_and_later_lines_win(cli, tmp_path):
    """#1, #7: the board's key=value parser (tinyseek_core parse_kv, used by perf::read_kv)
    and the PC's layout.read_kv give the same map."""
    from tinyseek.deploy.layout import read_kv

    path = tmp_path / "config.txt"
    path.write_bytes(b"\xef\xbb\xbfprompt=backpack\r\nwifi=off\r\n   wifi=sta\r\n\twifi_ssid = my net \r\n"
                     b"  # wifi_password=commented\r\nnoequals\r\n=nokey\r\nurl=http://x/?a=b\r\n\r\n")
    want = {"prompt": "backpack", "wifi": "sta", "wifi_ssid": "my net", "url": "http://x/?a=b"}
    assert run_kv(cli, path) == want
    assert read_kv(path) == want


@pytest.mark.parametrize("resample", ["pillow", "bilinear"])
@pytest.mark.parametrize("pad", [0, 255])
def test_pad_value_from_the_sidecar_is_used(cli, tmp_path, resample, pad):
    """#6: the plan carries the model's pad value; 114 stays the byte-identical default."""
    rng = np.random.default_rng(pad)
    rgb = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    default, _ = run_preprocess(cli, Image.fromarray(rgb), 192, -6, tmp_path, resample)
    assert np.array_equal(default, preprocess_rgb(rgb, 192, -6, resample)[0])
    # pad=114 given explicitly is the default output
    r = subprocess.run([str(cli), "preprocess", str(tmp_path / "in.rgb"), "400", "300", "192", "-6",
                        str(tmp_path / "out.bin"), resample, "pad=114"], check=True, capture_output=True)
    assert r.returncode == 0 and np.array_equal(np.fromfile(tmp_path / "out.bin", np.int8).reshape(192, 192, 3), default)
    subprocess.run([str(cli), "preprocess", str(tmp_path / "in.rgb"), "400", "300", "192", "-6",
                    str(tmp_path / "out.bin"), resample, f"pad={pad}"], check=True, capture_output=True)
    padded = np.fromfile(tmp_path / "out.bin", np.int8).reshape(192, 192, 3)
    ref, _ = preprocess_rgb(rgb, 192, -6, resample, pad_value=pad)
    assert np.array_equal(padded, ref)
    assert not np.array_equal(padded, default) and np.array_equal(padded[24:168], default[24:168])


def test_decode_clamps_degenerate_box_sizes(cli, tmp_path):
    """#8: with a large output exponent, exp(log_w) would overflow to inf; board and PC
    both clamp log_w/log_h at MAX_LOG_WH and give the same finite boxes."""
    from tinyseek.deploy.reference import MAX_LOG_WH

    _, lb = model_input(Image.new("RGB", (800, 600)), 192, -6)
    q = np.full((24, 24, 5), -128, np.int8)
    q[5, 7] = [127, 127, 100, 0, 0]  # exponent 3: logit 1016, log_w 1016, log_h 800
    q[15, 15] = [127, 1, -5, 0, 0]   # an ordinary box (log_w 8, log_h -40)
    ref = decode_output(q, 3, lb)
    got = run_decode(cli, q, 3, lb, tmp_path)
    assert len(ref) == len(got) == 2
    assert all(np.isfinite([d.x, d.y, d.w, d.h]).all() for d in ref + got)
    assert max(d.w for d in ref) == pytest.approx(np.float32(np.exp(np.float32(MAX_LOG_WH))) * 8 / lb.scale, rel=1e-5)
    for a, b in zip(sorted(ref, key=lambda d: d.w), sorted(got, key=lambda d: d.w)):
        assert abs(a.score - b.score) < 1e-6
        assert np.allclose([a.x, a.y, a.w, a.h], [b.x, b.y, b.w, b.h], rtol=1e-6, atol=1e-3)
