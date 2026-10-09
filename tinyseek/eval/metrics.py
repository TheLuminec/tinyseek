"""Detection metrics: AP@0.5 and expected calibration error of detection confidences.

``coco_ap50`` (pycocotools) is the number to report. ``match_detections`` and
``average_precision`` are a small self-contained reference used for ECE (which needs
per-detection true/false-positive labels) and for unit tests.

Detections are dicts with ``image_id``, ``bbox`` (xywh, original pixels), ``score``.
"""

from __future__ import annotations

import contextlib
import io
from collections import defaultdict

import numpy as np


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xywh boxes."""
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def match_detections(
    detections: list[dict], gt_boxes: dict[int, np.ndarray], iou_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy highest-score-first matching. Returns (scores, is_true_positive, num_gt)."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for d in detections:
        by_image[d["image_id"]].append(d)
    scores, tps = [], []
    for image_id, dets in by_image.items():
        dets = sorted(dets, key=lambda d: -d["score"])
        gts = gt_boxes.get(image_id, np.zeros((0, 4)))
        used = np.zeros(len(gts), dtype=bool)
        ious = iou_xywh(np.asarray([d["bbox"] for d in dets], dtype=np.float64), gts) if len(gts) else None
        for k, d in enumerate(dets):
            hit = False
            if ious is not None:
                cand = np.where(~used, ious[k], -1.0)
                j = int(cand.argmax())
                if cand[j] >= iou_threshold:
                    used[j] = hit = True
            scores.append(d["score"])
            tps.append(hit)
    num_gt = sum(len(g) for g in gt_boxes.values())
    return np.asarray(scores, dtype=np.float64), np.asarray(tps, dtype=bool), num_gt


def average_precision(scores: np.ndarray, is_tp: np.ndarray, num_gt: int) -> float:
    """Area under the interpolated precision-recall curve, sampled at 101 recall points (COCO style)."""
    if num_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    # Make precision monotonically non-increasing from the right.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += precision[idx] if idx < len(precision) else 0.0
    return ap / 101


def expected_calibration_error(scores: np.ndarray, is_tp: np.ndarray, num_bins: int = 10) -> float:
    """Detection ECE: per confidence bin, |precision - mean confidence|, weighted by bin size.

    This is the box-agnostic form of the detection calibration error of Kueppers et al.
    (2020): a detection is "correct" if it matches a ground-truth box at the IoU threshold.
    """
    if len(scores) == 0:
        return float("nan")
    edges = np.linspace(0, 1, num_bins + 1)
    bins = np.clip(np.digitize(scores, edges[1:-1]), 0, num_bins - 1)
    ece = 0.0
    for b in range(num_bins):
        sel = bins == b
        if sel.any():
            ece += sel.mean() * abs(is_tp[sel].mean() - scores[sel].mean())
    return float(ece)


def coco_ap50(gt_json: str, detections: list[dict], category_id: int) -> float:
    """Standard COCO AP at IoU 0.5 (pycocotools), restricted to one category."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        dt = gt.loadRes([dict(d, category_id=category_id) for d in detections])
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.catIds = [category_id]
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1])
