import json

import numpy as np
import pytest

from tinyseek.eval.metrics import (
    average_precision,
    coco_ap50,
    expected_calibration_error,
    iou_xywh,
    match_detections,
)


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5]], float)
    np.testing.assert_allclose(iou_xywh(a, b), [[1.0, 1 / 3, 0.0]])


def test_perfect_and_empty_ap():
    gt = {1: np.array([[0, 0, 10, 10]], float), 2: np.array([[5, 5, 10, 10]], float)}
    dets = [{"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 2, "bbox": [5, 5, 10, 10], "score": 0.8}]
    s, tp, n = match_detections(dets, gt)
    assert tp.all() and n == 2
    assert average_precision(s, tp, n) == pytest.approx(1.0)
    s, tp, n = match_detections([], gt)
    assert average_precision(s, tp, n) == 0.0


def test_duplicate_detection_is_false_positive():
    gt = {1: np.array([[0, 0, 10, 10]], float)}
    dets = [{"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}]
    _, tp, _ = match_detections(dets, gt)
    assert tp.tolist() == [True, False]


def test_ece():
    # Perfectly calibrated: in each bin, precision equals confidence.
    scores = np.array([0.25] * 4 + [0.75] * 4)
    tp = np.array([1, 0, 0, 0, 1, 1, 1, 0], bool)
    assert expected_calibration_error(scores, tp, 10) == pytest.approx(0.0)
    # Fully overconfident: score 0.9, always wrong.
    assert expected_calibration_error(np.full(5, 0.9), np.zeros(5, bool)) == pytest.approx(0.9)


def test_reference_ap_matches_pycocotools(tmp_path):
    rng = np.random.default_rng(0)
    images, anns, gt, dets = [], [], {}, []
    for i in range(1, 31):
        images.append({"id": i, "width": 200, "height": 200, "file_name": f"{i}.jpg"})
        boxes = []
        for _ in range(rng.integers(0, 3)):
            x, y = rng.uniform(0, 150, 2)
            w, h = rng.uniform(20, 50, 2)
            boxes.append([x, y, w, h])
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 47, "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": 0})
            # A noisy detection per GT plus some random false positives.
            jitter = rng.normal(0, 4, 4)
            dets.append({"image_id": i, "bbox": list(np.array([x, y, w, h]) + jitter), "score": float(rng.uniform(0.3, 1))})
        for _ in range(rng.integers(0, 2)):
            dets.append({"image_id": i, "bbox": list(rng.uniform(0, 150, 2)) + [30, 30], "score": float(rng.uniform(0, 0.7))})
        gt[i] = np.array(boxes, float).reshape(-1, 4)
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"images": images, "annotations": anns, "categories": [{"id": 47, "name": "cup"}]}))
    s, tp, n = match_detections(dets, gt)
    assert average_precision(s, tp, n) == pytest.approx(coco_ap50(str(path), dets, 47), abs=0.02)
