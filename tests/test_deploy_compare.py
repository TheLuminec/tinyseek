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


# The header and run.txt keys the firmware writes (firmware/perf_eval/main/perf_eval.cpp);
# both builds share them, the Arduino one differs only in the values.
BENCH_HEADER = ("run,board,build,idf_version,espdl_version,jpeg_decoder,cpu_mhz,model,model_bytes,input_size,"
                "input_exponent,output_exponent,internal_kb,stage,runs,warmup,mean_ms,min_ms,max_ms,std_ms,fps")
ARDUINO_ROW = ("run_001,freenove_esp32s3_cam_n16r8,arduino-esp32 3.3.12,v5.5.5,esp-dl bundled with core 3.3.12,"
               "esp32-camera fmt2rgb888 (ROM TJpgDec),240,px192,649728,192,-6,-4,0,inference,50,5,"
               "1.000,0.900,1.200,0.050,0.00")


def test_compare_reads_firmware_bench_and_run_files(tmp_path, capsys):
    from tinyseek.deploy.compare import print_report

    stage, res = _stage(tmp_path, board_out_delta=0, board_in_delta=0)
    (res / "bench.csv").write_text(BENCH_HEADER + "\n" + ARDUINO_ROW + "\n")
    (res / "run.txt").write_text("board=freenove_esp32s3_cam_n16r8\nbuild=arduino-esp32 3.3.12\n"
                                 "sdk_settings=dcache=32KB/32B icache=16KB spiram=80MHz sdk_opt=size\ncomplete=1\n")
    report = compare(stage, res)
    assert report["bench"][0]["build"] == "arduino-esp32 3.3.12"
    assert report["bench"][0]["stage"] == "inference"
    assert report["board"]["sdk_settings"].startswith("dcache=32KB")
    print_report(report)
    out = capsys.readouterr().out
    assert "build=arduino-esp32 3.3.12" in out and "inference" in out


# The firmware appends the model's I/O dtypes to every bench.csv row (int16 for w16a16).
BENCH_HEADER_IO = BENCH_HEADER + ",input_dtype,output_dtype"


def test_compare_int16_model(tmp_path):
    """A w16a16 model: int16 tensors on both sides, differences counted in int16 steps."""
    rng = np.random.default_rng(2)
    size, grid = 128, 16
    exp_in = rng.integers(-20000, 20000, size * size * 3).astype(np.int16)
    q = np.full((grid, grid, 5), -25000, np.int16)
    q[..., 1:3], q[..., 3:] = 8000, 2000
    q[4, 4, 0], q[10, 12, 0] = 12000, 6000
    stage = tmp_path / "stage"
    exp = stage / "expected" / "w16"
    exp.mkdir(parents=True)
    exp_in.tofile(exp / "c000_in.bin")
    q.tofile(exp / "c000_out.bin")
    lb = [0.16, 0, 16]
    (exp / "c000_float.json").write_text(json.dumps({"letterbox": lb, "detections": []}))
    model = {"name": "w16", "input_size": size, "stride": 8, "num_classes": 1, "output_exponent": -12,
             "score_threshold": 0.05, "max_detections": 20, "input_dtype": "INT16", "output_dtype": "INT16"}
    (stage / "stage.json").write_text(json.dumps({"models": [model], "cases": [{"id": "c000"}]}))
    res = tmp_path / "run_001"
    board = res / "parity" / "w16"
    board.mkdir(parents=True)
    b_out = q.copy()
    b_out[4, 4, 1] += 300  # 300 int16 steps of log_w
    from tinyseek.data.transforms import Letterbox

    dets = decode_output(b_out, -12, Letterbox(*lb))
    with (board / "dets.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DETS_HEADER)
        for path in ("rgb", "jpg"):
            exp_in.tofile(board / f"c000_{path}_in.bin")
            b_out.tofile(board / f"c000_{path}_out.bin")
            for r, d in enumerate(dets):
                w.writerow(["w16", "c000", path, r, d.x, d.y, d.w, d.h, d.score, d.cls])
    (res / "bench.csv").write_text(BENCH_HEADER_IO + "\n" + ARDUINO_ROW + ",INT16,INT16\n")
    report = compare(stage, res)
    s = report["models"]["w16"]["summary"]
    assert s["preprocess_exact"] == "1/1" and s["decode_identical"] == "2/2"
    assert s["model_output_exact"] == "0/1" and s["model_output_max_abs_lsb"] == 300
    assert len(dets) == 2 and report["bench"][0]["input_dtype"] == "INT16"


def test_compare_rectangular_model(tmp_path):
    """A staged 256x192 (width x height) model: stage.json has input_width/input_height, the
    tensors are 192 rows x 256 columns and the grid is 24 rows x 32 columns."""
    from tinyseek.config import size_fields
    from tinyseek.data.transforms import Letterbox

    rng = np.random.default_rng(3)
    w, h, gw, gh = 256, 192, 32, 24
    exp_in = rng.integers(-128, 127, w * h * 3).astype(np.int8)
    q = np.full((gh, gw, 5), -100, np.int8)
    q[..., 1:3], q[..., 3:] = 20, 8
    q[4, 30, 0], q[20, 2, 0] = 50, 30  # column 30 does not exist in a 24-wide grid
    stage = tmp_path / "stage"
    exp = stage / "expected" / "r256"
    exp.mkdir(parents=True)
    exp_in.tofile(exp / "c000_in.bin")
    q.tofile(exp / "c000_out.bin")
    lb = [0.32, 0, 0]
    (exp / "c000_float.json").write_text(json.dumps({"letterbox": lb, "detections": []}))
    model = {"name": "r256", **size_fields([w, h]), "stride": 8, "num_classes": 1, "output_exponent": -4,
             "score_threshold": 0.05, "max_detections": 20}
    assert "input_size" not in model
    (stage / "stage.json").write_text(json.dumps({"models": [model], "cases": [{"id": "c000"}]}))
    res = tmp_path / "run_001"
    board = res / "parity" / "r256"
    board.mkdir(parents=True)
    dets = decode_output(q, -4, Letterbox(*lb))
    assert len(dets) == 2 and max(d.x + d.w / 2 for d in dets) > 700  # in 800x600 frame pixels
    with (board / "dets.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(DETS_HEADER)
        for path in ("rgb", "jpg"):
            exp_in.tofile(board / f"c000_{path}_in.bin")
            q.tofile(board / f"c000_{path}_out.bin")
            for r, d in enumerate(dets):
                wr.writerow(["r256", "c000", path, r, d.x, d.y, d.w, d.h, d.score, d.cls])
    # The firmware writes a rectangular model's bench.csv input_size as WIDTHxHEIGHT.
    (res / "bench.csv").write_text(BENCH_HEADER_IO + "\n" + ARDUINO_ROW.replace(",192,-6,", ",256x192,-6,")
                                   + ",INT8,INT8\n")
    report = compare(stage, res)
    s = report["models"]["r256"]["summary"]
    assert s["preprocess_exact"] == "1/1" and s["model_output_exact"] == "1/1" and s["decode_identical"] == "2/2"
    assert s["rgb_dets_vs_sim_f1"] == 1.0
    assert report["bench"][0]["input_size"] == "256x192"
    # A board tensor of the square size is reported, not reshaped.
    exp_in[: 192 * 192 * 3].tofile(board / "c000_rgb_in.bin")
    assert compare(stage, res)["models"]["r256"]["cases"][0]["rgb"] == {"bad_size": [192 * 192 * 3, gw * gh * 5]}
