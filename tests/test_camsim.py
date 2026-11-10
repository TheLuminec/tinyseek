import numpy as np
from PIL import Image, ImageDraw

from tinyseek.generate.camsim import PRESETS, distort, distort_box, simulate


def test_box_moves_with_the_pixels():
    image = Image.new("RGB", (512, 512))
    box = [300.0, 60.0, 150.0, 120.0]
    ImageDraw.Draw(image).rectangle([300, 60, 450, 180], fill=(255, 255, 255))
    mask = np.asarray(distort(image, 0.35))[..., 0] > 128
    ys, xs = np.nonzero(mask)
    mapped = distort_box(box, 0.35, (512, 512))
    assert np.allclose(mapped, [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()], atol=2)


def test_no_distortion_is_identity_for_boxes():
    assert distort_box([10, 20, 30, 40], 0.0, (512, 512)) == [10, 20, 30, 40]


def test_simulate_matches_camera_brightness():
    rng = np.random.default_rng(0)
    bright = Image.new("RGB", (256, 256), (230, 220, 210))
    ImageDraw.Draw(bright).rectangle([60, 60, 200, 200], fill=(20, 120, 240))
    out, boxes = simulate(bright, [[60, 60, 140, 140]], PRESETS["camera"], rng)
    luma = np.asarray(out.convert("L"), dtype=np.float32).mean()
    assert 40 < luma < 90 and boxes[0] is not None
