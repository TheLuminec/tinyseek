"""Board/PC comparison logic in tinyseek.deploy.compare, on a synthetic staged run."""

from __future__ import annotations

import csv
import json

import numpy as np

from tinyseek.deploy.compare import compare, decode_identical, iou, match
from tinyseek.deploy.layout import DETS_HEADER
from tinyseek.deploy.reference import Detection, decode_output


def det(x, y, w, h, s=0.9):
    return Detection(x, y, w, h, s)


def test_iou_and_match():
    assert iou(det(0, 0, 10, 10), det(0, 0, 10, 10)) == 1.0
    assert abs(iou(det(0, 0, 10, 10), det(5, 0, 10, 10)) - 1 / 3) < 1e-9
    ref = [det(0, 0, 10, 10, 0.9), det(100, 100, 10, 10, 0.5)]
    got = [det(1, 0, 10, 10, 0.88), det(300, 300, 5, 5, 0.3)]
    m = match(ref, got)
    assert (m["matched"], m["ref"], m["board"]) == (1, 2, 2)
    assert m["f1"] == 0.5 and abs(m["max_score_diff"] - 0.02) < 1e-9
    assert match([], [])["f1"] == 1.0


def test_decode_identical_tolerates_float_noise_only():
    a = [det(1, 2, 3, 4, 0.5), det(5, 6, 7, 8, 0.4)]
    assert decode_identical(a, [det(1.0004, 2, 3, 4, 0.500001), det(5, 6, 7, 8, 0.4)])
    assert not decode_identical(a, a[:1])
    assert not decode_identical(a, [det(1.1, 2, 3, 4, 0.5), a[1]])


def _stage(tmp_path, board_out_delta: int, board_in_delta: int):
    """One model, one case; expected tensors, plus board files perturbed as asked."""
    rng = np.random.default_rng(1)
    size, grid = 128, 16
    exp_in = rng.integers(-128, 127, size * size * 3).astype(np.int8)
    q = np.full((grid, grid, 5), -100, np.int8)
    q[..., 1:3], q[..., 3:] = 20, 8
    q[4, 4, 0], q[10, 12, 0] = 50, 30
    stage = tmp_path / "stage"
    (stage / "expected" / "px128").mkdir(parents=True)
    exp_in.tofile(stage / "expected" / "px128" / "c000_in.bin")
    q.tofile(stage / "expected" / "px128" / "c000_out.bin")
    lb = [0.16, 0, 16]
    (stage / "expected" / "px128" / "c000_float.json").write_text(json.dumps({"letterbox": lb, "detections": []}))
    model = {"name": "px128", "input_size": size, "stride": 8, "num_classes": 1, "output_exponent": -4,
             "score_threshold": 0.05, "max_detections": 20}
    (stage / "stage.json").write_text(json.dumps({"models": [model], "cases": [{"id": "c000"}]}))

    res = tmp_path / "run_001"
    board = res / "parity" / "px128"
    board.mkdir(parents=True)
    b_in = exp_in.copy()
    b_in[:board_in_delta] += 1
    b_out = q.copy()
    b_out[4, 4, 1] += board_out_delta
    from tinyseek.data.transforms import Letterbox

    dets = decode_output(b_out, -4, Letterbox(*lb))
    with (board / "dets.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DETS_HEADER)
        for path in ("rgb", "jpg"):
            b_in.tofile(board / f"c000_{path}_in.bin")
            b_out.tofile(board / f"c000_{path}_out.bin")
            for r, d in enumerate(dets):
                w.writerow(["px128", "c000", path, r, d.x, d.y, d.w, d.h, d.score, d.cls])
    return stage, res


def test_compare_exact_board(tmp_path):
    stage, res = _stage(tmp_path, board_out_delta=0, board_in_delta=0)
    s = compare(stage, res)["models"]["px128"]["summary"]
    assert s["preprocess_exact"] == "1/1"
    assert s["model_output_exact"] == "1/1"
    assert s["decode_identical"] == "2/2"
    assert s["rgb_dets_vs_sim_f1"] == 1.0


def test_compare_reports_lsb_differences(tmp_path):
    stage, res = _stage(tmp_path, board_out_delta=2, board_in_delta=0)
    s = compare(stage, res)["models"]["px128"]["summary"]
    assert s["model_output_exact"] == "0/1" and s["model_output_max_abs_lsb"] == 2
    assert s["decode_identical"] == "2/2"  # decode is still checked on the board's own output
    assert s["rgb_dets_vs_sim_f1"] == 1.0  # a small log_w change keeps IoU >= 0.5


def test_compare_does_not_judge_model_when_inputs_differ(tmp_path):
    stage, res = _stage(tmp_path, board_out_delta=0, board_in_delta=5)
    s = compare(stage, res)["models"]["px128"]["summary"]
    assert s["preprocess_exact"] == "0/1"
    assert "model_output_exact" not in s
