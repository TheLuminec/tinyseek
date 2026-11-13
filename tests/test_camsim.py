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


def _scene() -> tuple[Image.Image, list[float]]:
    image = Image.new("RGB", (256, 192), (200, 190, 180))
    ImageDraw.Draw(image).rectangle([150, 30, 220, 100], fill=(30, 110, 230))
    return image, [150.0, 30.0, 70.0, 70.0]


def _pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def test_single_part_presets_change_only_their_part():
    image, box = _scene()
    src = _pixels(image)

    out, boxes = simulate(image, [box], PRESETS["only_distort"], np.random.default_rng(0))
    assert not np.allclose(boxes[0], box, atol=2)                       # geometry moved
    assert np.allclose(_pixels(out)[96, 128], src[96, 128], atol=1)      # centre colour untouched
    assert np.allclose(_pixels(out)[5, 128], (200, 190, 180), atol=1)    # background stays its colour

    out, boxes = simulate(image, [box], PRESETS["only_tone"], np.random.default_rng(0))
    assert boxes[0] == box
    assert _pixels(out).mean() < src.mean() - 40                         # pulled down to the camera's brightness
    blue = _pixels(out)[..., 2] - _pixels(out)[..., 0] > 20              # the square is still where it was
    ys, xs = np.nonzero(blue)
    assert abs(xs.min() - 150) <= 1 and abs(ys.min() - 30) <= 1

    out, boxes = simulate(image, [box], PRESETS["only_vignette"], np.random.default_rng(0))
    assert boxes[0] == box
    assert np.allclose(_pixels(out)[96, 128], src[96, 128], atol=2)      # centre kept
    assert _pixels(out)[0, 0].mean() < src[0, 0].mean() * 0.75           # corner darkened

    out, boxes = simulate(image, [box], PRESETS["only_noise"], np.random.default_rng(3))
    assert boxes[0] == box
    assert abs(_pixels(out).mean() - src.mean()) < 4                     # no tone shift
    assert not np.array_equal(_pixels(out), src)
