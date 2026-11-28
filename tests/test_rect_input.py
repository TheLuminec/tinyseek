"""Rectangular student inputs (``model.input_size: [width, height]``) in training and evaluation.

The size helper and its validation, letterboxing and augmentation, target encoding and
decoding, the model, the budget estimate, and prediction back to frame pixels. Square inputs
are pinned to what the code gave before rectangular ones existed (hashes from commit
40b4c70). The deployment side is in tests/test_deploy_reference.py and
tests/test_deploy_compare.py, the firmware C++ in tests/test_deploy_core.py and
tests/test_deploy_detector.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from tinyseek.config import (
    INPUT_MULTIPLE,
    check_input_size,
    fields_wh,
    input_wh,
    load_config,
    save_run_config,
    size_fields,
)
from tinyseek.data.dataset import DetectionDataset, Record, collate_eval
from tinyseek.data.transforms import augment, clip_boxes, letterbox, letterbox_geometry
from tinyseek.eval.evaluate import predict
from tinyseek.train.budget import measure
from tinyseek.train.loss import centernet_loss
from tinyseek.train.model import ExportWrapper, TinyCenterNet, build_model
from tinyseek.train.targets import decode, encode_targets

RECT = [(256, 192), (288, 224), (192, 256)]  # (width, height)


def sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()[:16]


def frame(seed: int, w: int, h: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


# --- the size helper and its validation --------------------------------------------------------

def test_input_wh_takes_an_int_or_a_width_height_pair():
    assert input_wh(192) == (192, 192)
    assert input_wh([256, 192]) == (256, 192)
    assert input_wh((288, 224)) == (288, 224)
    for bad in ([256], [256, 192, 3], "256x192", 192.0, [256.0, 192], None, True, 0, -32, [256, 0], {"w": 1}):
        with pytest.raises(ValueError, match="input size"):
            input_wh(bad)


def test_sizes_must_be_multiples_of_32():
    assert INPUT_MULTIPLE == 32
    for ok in (128, 160, 192, 256, 320, [256, 192], [288, 224], [320, 256], [192, 256]):
        assert check_input_size(ok) == input_wh(ok)
    for bad in (200, 144, [250, 192], [256, 200], [272, 192], [256, 208]):
        with pytest.raises(ValueError, match="multiples of 32"):
            check_input_size(bad)


@pytest.mark.parametrize("wh", [(256, 208), (272, 192), (240, 176), (160, 144)])
def test_the_model_really_needs_multiples_of_32(wh):
    """What the validation stands for: a side that is only a multiple of 16 breaks the fusion."""
    model = TinyCenterNet(pretrained=False).eval()
    with pytest.raises(RuntimeError), torch.no_grad():
        model(torch.zeros(1, 3, wh[1], wh[0]))


def test_config_accepts_a_list_and_rejects_bad_sizes_early(tmp_path):
    cfg = load_config("configs/base.yaml", ["model.input_size=[256,192]"])
    assert cfg["model"]["input_size"] == [256, 192] and input_wh(cfg["model"]["input_size"]) == (256, 192)
    assert load_config("configs/base.yaml", ["model.input_size=[288, 224]"])["model"]["input_size"] == [288, 224]
    assert load_config("configs/base.yaml")["model"]["input_size"] == 192  # the default stays an int
    for bad in ("model.input_size=[250,192]", "model.input_size=200", "model.input_size=256x192",
                "model.input_size=[256,192,3]"):
        with pytest.raises(ValueError, match="input"):
            load_config("configs/base.yaml", [bad])
    # A run's saved config.yaml keeps either form.
    for size in (192, [256, 192]):
        run = tmp_path / str(size)
        save_run_config(load_config("configs/base.yaml", [f"model.input_size={size}"]), run)
        assert load_config(run / "config.yaml")["model"]["input_size"] == size
    # A config without a model section (QAT settings, generation) loads as before.
    assert "model" not in load_config("configs/qat.yaml")


def test_size_fields_keep_input_size_for_squares():
    assert size_fields(192) == {"input_size": 192}
    assert size_fields([192, 192]) == {"input_size": 192}
    assert size_fields([256, 192]) == {"input_width": 256, "input_height": 192}
    assert fields_wh({"input_size": 192}) == (192, 192)
    assert fields_wh({"input_width": 256, "input_height": 192}) == (256, 192)
    assert fields_wh({"input_size": "128"}) == (128, 128)  # key=value files hold strings
    assert fields_wh({"input_width": "288", "input_height": "224"}) == (288, 224)
    with pytest.raises(KeyError):
        fields_wh({"input_width": 256})
    for size in (128, [256, 192], (288, 224)):
        assert fields_wh(size_fields(size)) == input_wh(size)


# --- letterbox and augmentation ----------------------------------------------------------------

def test_square_letterbox_augment_and_targets_are_unchanged():
    """Pinned with the square-only code (commit 40b4c70): a zoomed and shifted letterbox,
    six augmentation draws (no JPEG step, which depends on the libjpeg build) and targets."""
    rgb = frame(7, 640, 480)
    canvas, lb = letterbox(Image.fromarray(rgb), 192, zoom=1.3, shift=(0.25, 0.8))
    assert (sha(np.asarray(canvas)), lb.scale, lb.pad_x, lb.pad_y) == ("dbac13a9465470db", 0.39, -14, 4)
    boxes = np.array([[100.0, 100, 400, 400], [10.0, 20, 60, 90], [500.0, 300, 640, 480]], np.float32)
    rng = np.random.default_rng(11)
    cfg = {"hflip": 0.5, "scale_jitter": [0.6, 1.4], "color_jitter": 0.3, "sensor_noise": False}
    parts = []
    for _ in range(6):
        img, b = augment(Image.fromarray(rgb), boxes, 192, cfg, rng)
        parts += [np.asarray(img), b]
    assert sha(*parts) == "94d0756308524bb0"
    t = encode_targets(np.array([[10.0, 20.0, 50.0, 90.0], [70.0, 64.0, 120.0, 100.0], [150.0, 3.0, 191.0, 30.0]],
                                np.float32), np.zeros(3, np.int64), 1, 192, 8)
    assert sha(*[t[k] for k in ("heatmap", "index", "mask", "size", "offset")]) == "e8e8ae5d85b4756c"


@pytest.mark.parametrize("wh", [(640, 480), (800, 600), (333, 217), (64, 200), (1, 1), (500, 500), (4000, 3)])
@pytest.mark.parametrize("size", [128, 192, 256])
def test_square_geometry_is_the_long_side_rule(wh, size):
    """For a square input the general rule reduces to the earlier one, to the last bit."""
    w, h = wh
    for zoom in (1.0, 0.6, 1.3, 1.4):
        scale = size * zoom / max(w, h)  # the square-only formula
        want = (scale, max(1, round(w * scale)), max(1, round(h * scale)))
        got = letterbox_geometry(w, h, size, zoom, (0.25, 0.8))
        assert got[:3] == want
        assert got[3:] == (round((size - want[1]) * 0.25), round((size - want[2]) * 0.8))
        assert letterbox_geometry(w, h, (size, size), zoom, (0.25, 0.8)) == got


@pytest.mark.parametrize("size,frame_wh,scale,pad,resized", [
    ((256, 192), (800, 600), 0.32, (0, 0), (256, 192)),    # 4:3 into 4:3: no padding
    ((256, 192), (640, 480), 0.4, (0, 0), (256, 192)),
    ((288, 224), (800, 600), 0.36, (0, 4), (288, 216)),    # 9:7 input: 4 rows above and below
    ((320, 256), (800, 600), 0.4, (0, 8), (320, 240)),
    ((256, 192), (600, 800), 0.24, (56, 0), (144, 192)),   # a portrait frame: bars left and right
    ((256, 192), (500, 500), 0.384, (32, 0), (192, 192)),
    ((192, 256), (800, 600), 0.24, (0, 56), (192, 144)),   # a tall input
    ((256, 192), (1600, 600), 0.16, (0, 48), (256, 96)),   # a wide frame: bars above and below
    ((256, 192), (256, 192), 1.0, (0, 0), (256, 192)),     # already the input size: copied as is
])
def test_rect_letterbox_geometry(size, frame_wh, scale, pad, resized):
    got = letterbox_geometry(*frame_wh, size)
    assert got == (scale, *resized, *pad)
    canvas, lb = letterbox(Image.fromarray(frame(1, *frame_wh)), size)
    assert canvas.size == size and (lb.scale, lb.pad_x, lb.pad_y) == (scale, *pad)
    arr = np.asarray(canvas)
    inside = np.zeros(arr.shape[:2], bool)
    inside[pad[1]:pad[1] + resized[1], pad[0]:pad[0] + resized[0]] = True
    assert (arr[~inside] == 114).all()  # the same padding value outside the picture
    if frame_wh == resized:
        assert np.array_equal(arr, frame(1, *frame_wh))


@pytest.mark.parametrize("size", RECT)
@pytest.mark.parametrize("frame_wh", [(800, 600), (640, 480), (480, 640), (333, 217)])
def test_rect_box_mapping_round_trip(size, frame_wh):
    """Frame boxes -> input pixels -> frame boxes, and the picture's corners land on the
    picture's corners in the input."""
    w, h = frame_wh
    _, lb = letterbox(Image.new("RGB", frame_wh), size)
    boxes = np.array([[0.0, 0.0, w, h], [10.0, 20.0, 110.0, 220.0], [w - 50.0, h - 40.0, w, h]], np.float32)
    inp = lb.forward(boxes)
    np.testing.assert_allclose(lb.inverse(inp), boxes, rtol=1e-5, atol=1e-3)
    _, nw, nh, pad_x, pad_y = letterbox_geometry(w, h, size)
    np.testing.assert_allclose(inp[0], [pad_x, pad_y, pad_x + w * lb.scale, pad_y + h * lb.scale], atol=1e-3)
    assert abs(pad_x + w * lb.scale - (pad_x + nw)) <= 0.5 and abs(pad_y + h * lb.scale - (pad_y + nh)) <= 0.5
    assert (inp[:, [0, 2]] >= -1e-3).all() and (inp[:, [0, 2]] <= size[0] + 1e-3).all()
    assert (inp[:, [1, 3]] >= -1e-3).all() and (inp[:, [1, 3]] <= size[1] + 1e-3).all()


def test_rect_clip_and_augment_keep_boxes_on_the_canvas():
    boxes = np.array([[-90.0, 0, 10, 100], [200.0, 100, 250, 150], [200.0, 150, 300, 260]], np.float32)
    # x clips at the width (256), y at the height (192); the third box keeps 42 of 110 rows
    np.testing.assert_allclose(clip_boxes(boxes, (256, 192)), [[200, 100, 250, 150]])
    np.testing.assert_allclose(clip_boxes(boxes, (256, 288)), [[200, 100, 250, 150], [200, 150, 256, 260]])
    rng = np.random.default_rng(0)
    img = Image.fromarray(frame(3, 800, 600))
    cfg = {"hflip": 0.5, "scale_jitter": [0.6, 1.4], "color_jitter": 0.3, "sensor_noise": True}
    kept = 0
    for _ in range(40):
        out, b = augment(img, np.array([[100.0, 100, 400, 400], [700.0, 50, 790, 120]]), (256, 192), cfg, rng)
        assert out.size == (256, 192)
        assert (b[:, [0, 2]] >= 0).all() and (b[:, [0, 2]] <= 256).all()
        assert (b[:, [1, 3]] >= 0).all() and (b[:, [1, 3]] <= 192).all()
        kept += len(b)
    assert kept > 20


def test_rect_hflip_mirrors_about_the_width():
    img = Image.fromarray(frame(4, 800, 600))
    box = np.array([[80.0, 60.0, 240.0, 300.0]])
    cfg = {"hflip": 1.0, "scale_jitter": [1.0, 1.0]}
    plain, _ = letterbox(img, (256, 192))
    for seed in range(5):  # at zoom 1 a 4:3 frame fills the input, wherever the random shift puts it
        out, b = augment(img, box, (256, 192), cfg, np.random.default_rng(seed))
        assert np.array_equal(np.asarray(out), np.asarray(plain)[:, ::-1])
        np.testing.assert_allclose(b, [[256 - 240 * 0.32, 60 * 0.32, 256 - 80 * 0.32, 300 * 0.32]], atol=1e-3)


# --- targets, model, loss ----------------------------------------------------------------------

@pytest.mark.parametrize("size", RECT)
def test_rect_encode_decode_roundtrip(size):
    w, h = size
    gw, gh = w // 8, h // 8
    # One box in each far corner region, so a swapped axis would clip or misplace it.
    boxes = np.array([[10.0, 20.0, 50.0, 90.0], [w - 60.0, h - 50.0, w - 8.0, h - 6.0], [w - 40.0, 4.0, w - 2.0, 30.0],
                      [6.0, h - 30.0, 40.0, h - 3.0]], dtype=np.float32)
    t = encode_targets(boxes, np.zeros(len(boxes), np.int64), num_classes=1, input_size=list(size), stride=8)
    assert t["heatmap"].shape == (1, gh, gw)
    assert t["mask"].sum() == len(boxes) and t["heatmap"].max() == 1.0
    assert (t["index"][: len(boxes)] < gw * gh).all()
    # index is row * grid_width + column: the peak of each object is where the heatmap is 1
    ones = set(np.flatnonzero(t["heatmap"][0].reshape(-1) == 1.0).tolist())
    assert ones == set(t["index"][: len(boxes)].tolist())

    heat = torch.full((1, 1, gh, gw), logit(1e-3))
    size_t = torch.zeros(1, 2, gh * gw)
    offset = torch.zeros(1, 2, gh * gw)
    for k in range(len(boxes)):
        idx = int(t["index"][k])
        heat.view(-1)[idx] = logit(0.9)
        size_t[0, :, idx] = torch.from_numpy(t["size"][k])
        offset[0, :, idx] = torch.from_numpy(t["offset"][k])
    out, scores, classes = decode(heat, size_t.view(1, 2, gh, gw), offset.view(1, 2, gh, gw), stride=8, top_k=len(boxes))
    np.testing.assert_allclose(sorted(out[0].numpy().tolist()), sorted(boxes.tolist()), atol=1e-3)
    assert torch.allclose(scores, torch.full((1, len(boxes)), 0.9), atol=1e-6) and (classes == 0).all()


@pytest.mark.parametrize("size", RECT)
def test_rect_model_shapes_loss_and_export_wrapper(size):
    w, h = size
    model = TinyCenterNet(pretrained=False)
    out = model(torch.randn(2, 3, h, w))  # NCHW: height first
    assert out["heatmap"].shape == (2, 1, h // 8, w // 8)
    assert out["size"].shape == out["offset"].shape == (2, 2, h // 8, w // 8)
    t = encode_targets(np.array([[10.0, 20.0, w - 20.0, h - 30.0]], np.float32), np.zeros(1, np.int64), 1, size, 8)
    batch = {k: torch.from_numpy(np.stack([v, v])) for k, v in t.items()}
    loss, parts = centernet_loss(out, batch)
    loss.backward()
    assert np.isfinite(loss.item()) and set(parts) == {"heatmap", "size", "offset"}
    assert ExportWrapper(model.eval())(torch.zeros(1, 3, h, w)).shape == (1, 5, h // 8, w // 8)


def test_budget_counts_the_rectangle():
    cfg = load_config("configs/base.yaml")
    model = build_model(dict(cfg["model"], pretrained=False))
    sq192, sq256 = measure(model, 192), measure(model, 256)
    rect = measure(model, [256, 192])
    assert sq192.input_size == 192 and rect.input_size == [256, 192]
    assert rect.macs == measure(model, [192, 256]).macs  # every layer is a stride-aligned convolution
    # 256x192 has 3/4 of the pixels of 256x256 and 4/3 of 192x192: so do MACs and activations.
    assert rect.macs * 4 == sq256.macs * 3 and rect.macs * 3 == sq192.macs * 4
    assert rect.peak_activation_bytes_int8 * 4 == sq256.peak_activation_bytes_int8 * 3
    assert rect.params == sq192.params and rect.weight_bytes_int8 == sq192.weight_bytes_int8
    assert rect.fits and measure(model, [288, 224]).fits
    assert measure(model, (256, 192)).macs == rect.macs
    with pytest.raises(ValueError, match="multiples of 32"):
        measure(model, [250, 192])


# --- dataset and prediction back to frame pixels -----------------------------------------------

def _photo(tmp_path, name: str, wh: tuple[int, int], seed: int) -> Record:
    path = tmp_path / name
    Image.fromarray(frame(seed, *wh)).save(path, quality=90)
    return Record(path, seed, np.array([[wh[0] * 0.6, wh[1] * 0.2, wh[0] * 0.9, wh[1] * 0.7]], np.float32))


@pytest.mark.parametrize("size", [(256, 192), [288, 224]])
def test_rect_dataset_shapes(tmp_path, size):
    w, h = size
    recs = [_photo(tmp_path, "a.jpg", (800, 600), 1), _photo(tmp_path, "b.jpg", (480, 640), 2)]
    train = DetectionDataset(recs, size, 8, train=True, augment_cfg={"hflip": 0.5, "scale_jitter": [0.6, 1.4]})
    item = train[0]
    assert item["image"].shape == (3, h, w) and item["heatmap"].shape == (1, h // 8, w // 8)
    assert int(item["index"].max()) < (h // 8) * (w // 8)
    batch = torch.utils.data.default_collate([train[0], train[1]])
    assert batch["image"].shape == (2, 3, h, w)
    ev = collate_eval([DetectionDataset(recs, size, 8, train=False)[i] for i in range(2)])
    assert ev["image"].shape == (2, 3, h, w)
    assert [lb.scale for lb in ev["letterbox"]] == [min(w / 800, h / 600), min(w / 480, h / 640)]


class PlantedModel:
    """Stands in for the student: one confident object per image at a fixed input-pixel box."""

    stride = 8

    def __init__(self, box: np.ndarray, wh: tuple[int, int]) -> None:
        self.t = encode_targets(box[None].astype(np.float32), np.zeros(1, np.int64), 1, wh, 8)
        self.gw, self.gh = wh[0] // 8, wh[1] // 8

    def eval(self) -> PlantedModel:
        return self

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        assert x.shape[1:] == (3, self.gh * 8, self.gw * 8)
        idx = int(self.t["index"][0])
        heat = torch.full((b, 1, self.gh * self.gw), logit(1e-3))
        size, offset = torch.zeros(b, 2, self.gh * self.gw), torch.zeros(b, 2, self.gh * self.gw)
        heat[:, 0, idx] = logit(0.9)
        size[:, :, idx] = torch.from_numpy(self.t["size"][0])
        offset[:, :, idx] = torch.from_numpy(self.t["offset"][0])
        shape = (b, -1, self.gh, self.gw)
        return {"heatmap": heat.view(shape), "size": size.view(shape), "offset": offset.view(shape)}


@pytest.mark.parametrize("size", [(256, 192), (288, 224), 192])
@pytest.mark.parametrize("frame_wh", [(800, 600), (480, 640)])
def test_predict_maps_boxes_back_to_frame_pixels(tmp_path, size, frame_wh):
    """predict() with a model that puts an object at the letterboxed place of a known frame box."""
    rec = _photo(tmp_path, "a.jpg", frame_wh, 1)
    _, lb = letterbox(Image.new("RGB", frame_wh), size)
    dets = predict(PlantedModel(lb.forward(rec.boxes)[0], input_wh(size)), [rec], size, torch.device("cpu"),
                   num_workers=0)
    assert len(dets) == 1 and abs(dets[0]["score"] - 0.9) < 1e-6
    x, y, w, h = dets[0]["bbox"]
    np.testing.assert_allclose([x, y, x + w, y + h], rec.boxes[0], atol=0.05)


