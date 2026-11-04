import numpy as np
from PIL import Image

from tinyseek.config import fill, load_config
from tinyseek.data.transforms import augment, clip_boxes, letterbox
from tinyseek.train.budget import measure
from tinyseek.train.model import build_model


def test_condition_configs_resolve():
    for name in ("real_only", "teacher_real", "synth_only", "synth_real"):
        cfg = load_config(f"configs/{name}.yaml", ["prompt=backpack", "seed=3"])
        assert cfg["condition"] == name and cfg["seed"] == 3
        assert cfg["model"]["input_size"] == 128  # inherited from base
        for src in cfg["train"]["sources"]:
            path = fill(src["ann"], cfg)
            assert "{" not in path and "backpack" in path
    synth_real = load_config("configs/synth_real.yaml")
    assert len(synth_real["train"]["sources"]) == 2


def test_letterbox_inverse():
    img = Image.new("RGB", (640, 480))
    canvas, lb = letterbox(img, 128)
    assert canvas.size == (128, 128) and lb.pad_x == 0 and lb.pad_y == 16
    boxes = np.array([[100.0, 50.0, 300.0, 400.0]])
    np.testing.assert_allclose(lb.inverse(lb.forward(boxes)), boxes, rtol=1e-5)


def test_clip_drops_mostly_hidden_boxes():
    boxes = np.array([[-90.0, 0, 10, 100], [10.0, 10, 60, 60]])
    np.testing.assert_allclose(clip_boxes(boxes, 128), [[10, 10, 60, 60]])


def test_augment_keeps_boxes_on_canvas():
    rng = np.random.default_rng(0)
    img = Image.new("RGB", (640, 480), (200, 50, 50))
    cfg = {"hflip": 0.5, "scale_jitter": [0.6, 1.4], "color_jitter": 0.3, "sensor_noise": True}
    for _ in range(20):
        out, boxes = augment(img, np.array([[100.0, 100, 400, 400]]), 128, cfg, rng)
        assert out.size == (128, 128)
        assert ((boxes >= 0) & (boxes <= 128)).all()


def test_default_student_fits_esp32s3():
    cfg = load_config("configs/base.yaml")
    budget = measure(build_model(dict(cfg["model"], pretrained=False)), cfg["model"]["input_size"])
    assert budget.fits, budget
