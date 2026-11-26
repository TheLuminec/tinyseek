"""The live detector's host-side pieces: its portable C++ logic, card staging and replay scoring.

firmware/detector/src/detector_logic.cpp (settings, prompt names, JSON lines) is built with
g++ through firmware/detector/host_test/logic_cli.cpp. The staging and scoring tests use a
fake run directory and synthetic JPEGs, so they need no data, runs or GPU.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tinyseek.deploy import detector_card, replay
from tinyseek.deploy.layout import read_kv

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "firmware" / "detector"

INFO = """graph main_graph (
  %input.1[INT8, 1x192x192x3], exponents: [-6]
) initializers (
) {
  %641 = Conv[group = 1](%input.1)
  return %641
}

%641[INT8, 1x24x24x5], exponents: [-4]
"""


@pytest.fixture(autouse=True)
def _cache(tmp_path_factory, monkeypatch):
    """Build the host TJpgDec (used by the staging checks) under pytest's temp, not runs/."""
    monkeypatch.setenv("TINYSEEK_BUILD_CACHE", str(tmp_path_factory.getbasetemp() / "build"))


@pytest.fixture(scope="module")
def cli(tmp_path_factory) -> Path:
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    out = tmp_path_factory.mktemp("detector") / "logic_cli"
    subprocess.run(["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", str(DET / "src" / "detector_logic.cpp"),
                    str(DET / "host_test" / "logic_cli.cpp"), "-o", str(out)], check=True)
    return out


def run(cli: Path, *args: str) -> list[str]:
    return subprocess.run([str(cli), *args], capture_output=True, text=True, check=True).stdout.splitlines()


def settings(cli: Path, tmp: Path, text: str) -> tuple[dict, list[str]]:
    (tmp / "config.txt").write_text(text)
    lines = run(cli, "settings", str(tmp / "config.txt"))
    return json.loads(lines[0]), lines[1:]


def test_settings_defaults_are_the_measured_path(cli, tmp_path):
    s, warnings = settings(cli, tmp_path, "")
    assert warnings == []
    assert s == {"prompt": "", "source": "camera", "decode_scale": 0, "resize": "pillow", "decoder": "tjpgd",
                 "multicore": 0, "internal_kb": 0, "pipeline": 0, "led": "both", "print_min_score": 0.1,
                 "wifi": "off", "http_port": 80, "replay_dir": "/sdcard/tinyseek/replay", "replay_limit": 0}


def test_settings_parse_every_switch(cli, tmp_path):
    lines = ["# comment", "prompt= Cup ", "source=replay", "decode_scale=2", "resize=bilinear", "multicore=1",
             "internal_kb=48", "pipeline=on", "led=external", "print_min_score=0.25", "wifi=sta", "wifi_ssid=net",
             "wifi_password=secret-value", "http_port=8080", "replay_limit=10"]
    s, warnings = settings(cli, tmp_path, "".join(f"{line}\n" for line in lines))
    assert warnings == []
    assert (s["prompt"], s["source"], s["decode_scale"], s["resize"], s["multicore"], s["internal_kb"],
            s["pipeline"], s["led"], s["print_min_score"], s["wifi"], s["wifi_ssid"], s["http_port"],
            s["replay_limit"]) == ("cup", "replay", 2, "bilinear", 1, 48, 1, "external", 0.25, "sta", "net", 8080, 10)
    assert "secret-value" not in json.dumps(s)


def test_settings_reject_bad_values_and_unsafe_wifi(cli, tmp_path):
    s, warnings = settings(cli, tmp_path, "decode_scale=5\nresize=nearest\nfoo=1\nwifi=ap\nap_password=short\n"
                                          "decoder=esp_new_jpeg\n")
    assert s["decode_scale"] == 0 and s["resize"] == "pillow" and s["wifi"] == "off"
    assert s["decoder"] == "esp_new_jpeg"
    text = "\n".join(warnings)
    for needle in ("decode_scale=5", "resize=nearest", "unknown key foo", "ap_password"):
        assert needle in text
    s, warnings = settings(cli, tmp_path, "decoder=esp_new_jpeg\ndecode_scale=1\nwifi=sta\n")
    assert s["decode_scale"] == 0 and s["wifi"] == "off"
    assert any("full size only" in w for w in warnings) and any("wifi_ssid" in w for w in warnings)


@pytest.mark.parametrize("typed,name,ok", [(" Back Pack ", "back_pack", 1), ("cup", "cup", 1), ("tea-cup", "tea_cup", 1),
                                           ("cup!", "cup!", 0), ("", "", 0), ("a" * 33, "a" * 33, 0)])
def test_prompt_names(cli, typed, name, ok):
    out = run(cli, "prompt", typed)
    assert out == [name, str(ok)] if name else out[-1] == str(ok)


def test_with_prompt_replaces_or_appends(cli, tmp_path):
    cfg = tmp_path / "c.txt"
    cfg.write_bytes(b"a=1\r\nprompt=backpack\r\n# wifi=sta\nb=2\n")
    out = subprocess.run([str(cli), "with_prompt", str(cfg), "cup"], capture_output=True, check=True).stdout
    assert out == b"a=1\r\nprompt=cup\n# wifi=sta\nb=2\n"  # other lines kept byte for byte
    cfg.write_text("a=1\n")
    assert run(cli, "with_prompt", str(cfg), "cup") == ["a=1", "prompt=cup"]


def test_frame_json_lines(cli):
    frame, rep, err = (json.loads(line) for line in run(cli, "frame"))
    assert frame["type"] == "frame" and frame["n"] == 7 and frame["prompt"] == 'back"pack'
    assert frame["hit"] is True and frame["ndet"] == 2 and len(frame["dets"]) == 1  # 0.07 < print_min_score
    assert frame["dets"][0] == {"s": 0.91, "x": 100.2, "y": 200.5, "w": 50.0, "h": 80.1}
    assert set(frame["t"]) == {"cap", "dec", "pre", "inf", "post", "lat", "period"}
    assert rep["file"] == "00001.jpg" and len(rep["dets"]) == 2 and rep["dets"][0]["x"] == 100.25
    assert err["ok"] is False and err["error"] == "JPEG decode failed" and err["dets"] == [] and err["hit"] is False


# --- staging -----------------------------------------------------------------------------------

def fake_run(path: Path, prompt: str = "backpack", meta_extra: dict | None = None) -> Path:
    path.mkdir(parents=True)
    (path / "model.espdl").write_bytes(b"EDL2" + bytes(range(64)))
    meta = {"prompt": prompt, "input_size": 192, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "stride": 8, "num_classes": 1, **(meta_extra or {})}
    (path / "model_meta.json").write_text(json.dumps(meta))
    (path / "model.info").write_text(INFO)
    (path / "config.yaml").write_text(f"prompt: {prompt}\neval:\n  score_threshold: 0.05\n  max_detections: 20\n")
    return path


def save_jpeg(path: Path, mode: str = "RGB", progressive: bool = False, size=(800, 600)) -> None:
    rng = np.random.default_rng(len(path.name))
    arr = rng.integers(60, 200, (size[1] // 32 + 1, size[0] // 32 + 1, 3), dtype=np.uint8)
    img = Image.fromarray(arr).resize(size).convert(mode)
    img.save(path, "JPEG", quality=85, progressive=progressive)


def replay_set(tmp: Path) -> tuple[Path, Path]:
    images = tmp / "images"
    (images / "s1").mkdir(parents=True)
    save_jpeg(images / "s1" / "a.jpg")
    save_jpeg(images / "s1" / "b.jpg")
    save_jpeg(images / "s1" / "prog.jpg", progressive=True)
    save_jpeg(images / "s1" / "gray.jpg", mode="L")
    coco = {
        "images": [{"id": 11, "file_name": "s1/a.jpg", "width": 800, "height": 600},
                   {"id": 12, "file_name": "s1/prog.jpg", "width": 800, "height": 600},
                   {"id": 13, "file_name": "s1/b.jpg", "width": 800, "height": 600},
                   {"id": 14, "file_name": "s1/gray.jpg", "width": 800, "height": 600}],
        "annotations": [{"id": 1, "image_id": 11, "category_id": 27, "bbox": [100, 120, 80, 90], "area": 7200,
                         "iscrowd": 0},
                        {"id": 2, "image_id": 12, "category_id": 27, "bbox": [10, 10, 50, 50], "area": 2500,
                         "iscrowd": 0}],
        "categories": [{"id": 27, "name": "backpack"}],
    }
    ann = tmp / "labels.json"
    ann.write_text(json.dumps(coco))
    return ann, images


def test_stage_card_layout_thresholds_and_replay(tmp_path, cli):
    r1 = fake_run(tmp_path / "runs" / "bp")
    r2 = fake_run(tmp_path / "runs" / "cup", "cup", {"detect_threshold": 0.45})
    ann, images = replay_set(tmp_path)
    out = tmp_path / "card"
    m = detector_card.stage([(None, r1), ("mug", r2)], out, None, {"backpack": 0.35}, None,
                            {"decode_scale": "1", "resize": "bilinear"}, ann, images, 0, 27)
    card = out / "sdcard" / "tinyseek"
    assert sorted(p.name for p in (card / "models").iterdir()) == ["backpack.espdl", "backpack.txt", "mug.espdl",
                                                                   "mug.txt"]
    bp, mug = read_kv(card / "models" / "backpack.txt"), read_kv(card / "models" / "mug.txt")
    assert (bp["detect_threshold"], bp["detect_threshold_source"]) == ("0.35", "cli")
    assert (mug["detect_threshold"], mug["detect_threshold_source"]) == ("0.45", "model_meta")
    assert (bp["input_exponent"], bp["output_exponent"], bp["score_threshold"]) == ("-6", "-4", "0.05")
    # The C++ sidecar parser reads what the stager wrote.
    meta = dict(line.split("=", 1) for line in run(cli, "meta", str(card / "models" / "backpack.txt")))
    assert (meta["prompt"], meta["input_size"], meta["has_exponents"], meta["detect_threshold"]) == \
        ("backpack", "192", "1", "0.35")
    # ...and the firmware parses the staged config.txt without a single warning.
    lines = run(cli, "settings", str(card / "config.txt"))
    assert lines[1:] == []
    s = json.loads(lines[0])
    assert (s["prompt"], s["decode_scale"], s["resize"], s["wifi"]) == ("backpack", 1, "bilinear", "off")
    # Replay: baseline RGB JPEGs copied in label order, the rest skipped with a reason.
    assert sorted(p.name for p in (card / "replay").iterdir()) == ["00000.jpg", "00001.jpg"]
    rep = m["replay"]
    assert [(f["file"], f["image_id"]) for f in rep["files"]] == [("00000.jpg", 11), ("00001.jpg", 13)]
    assert {s["image"] for s in rep["skipped"]} == {"s1/prog.jpg", "s1/gray.jpg"}
    labels = json.loads((out / "replay_labels.json").read_text())
    assert [i["id"] for i in labels["images"]] == [11, 13] and [a["image_id"] for a in labels["annotations"]] == [11]
    assert json.loads((out / "card.json").read_text())["replay"]["category_id"] == 27


def test_stage_refuses_credentials_and_bad_names(tmp_path):
    with pytest.raises(SystemExit):
        detector_card.parse_sets(["wifi_password=x"])
    with pytest.raises(SystemExit):
        detector_card.parse_sets(["nonsense=1"])
    r = fake_run(tmp_path / "r", "Back Pack")
    with pytest.raises(SystemExit):
        detector_card.stage([(None, r)], tmp_path / "out", None, {}, None, {}, None, None, 0, None)
    a, b = fake_run(tmp_path / "a"), fake_run(tmp_path / "b")
    with pytest.raises(SystemExit):
        detector_card.stage([(None, a), (None, b)], tmp_path / "out", None, {}, None, {}, None, None, 0, None)
    assert detector_card.parse_thresholds(["0.4", "cup=0.5"]) == (0.4, {"cup": 0.5})
    assert detector_card.parse_model_arg("cup=runs/x") == ("cup", Path("runs/x"))
    assert detector_card.parse_model_arg("runs/x=y") == (None, Path("runs/x=y"))


# --- replay scoring ----------------------------------------------------------------------------

def test_replay_score(tmp_path, cli):
    r1 = fake_run(tmp_path / "runs" / "bp")
    ann, images = replay_set(tmp_path)
    out = tmp_path / "card"
    detector_card.stage([(None, r1)], out, None, {}, None, {"source": "replay"}, ann, images, 0, 27)
    results = tmp_path / "run_001"
    results.mkdir()
    t = {"cap": 30.0, "dec": 210.0, "pre": 40.0, "inf": 400.0, "post": 0.4, "lat": 680.0, "period": 0}
    rows = [
        {"file": "00000.jpg", "w": 800, "h": 600, "ok": True, "t": t,
         "dets": [{"s": 0.9, "x": 101, "y": 119, "w": 80, "h": 90, "c": 0},
                  {"s": 0.2, "x": 500, "y": 400, "w": 20, "h": 20, "c": 0}]},
        {"file": "00001.jpg", "w": 800, "h": 600, "ok": False, "error": "JPEG decode failed", "t": t, "dets": []},
    ]
    sha = read_kv(out / "sdcard" / "tinyseek" / "models" / "backpack.txt")["espdl_sha256"]
    (results / "dets.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"file":"000')
    (results / "run.txt").write_text(f"prompt=backpack\nmodel_sha256={sha}\nsettings={{\"decode_scale\":1}}\n"
                                     "complete=1\n")
    s = replay.score(out, results)
    assert s["ap50"] == pytest.approx(1.0)
    assert s["num_images"] == 2 and s["num_gt"] == 1 and s["num_detections"] == 2
    assert s["failed"] == [{"file": "00001.jpg", "error": "JPEG decode failed"}] and s["missing"] == []
    assert s["complete"] and s["settings"] == {"decode_scale": 1} and s["warnings"] == []
    assert s["mean_ms"]["inf"] == 400.0
    # The firmware's own replay line (detector_logic replay_json) is what the scorer parses.
    line = run(cli, "frame")[1]
    (results / "dets.jsonl").write_text(line.replace("00001.jpg", "00000.jpg") + "\n")
    s = replay.score(out, results)
    assert s["num_detections"] == 2 and s["missing"] == ["00001.jpg"] and s["warnings"]


def test_replay_score_rejects_unknown_partial_lines(tmp_path):
    (tmp_path / "dets.jsonl").write_text('{"file": "x"\n{"file": "y", "dets": []}\n')
    with pytest.raises(json.JSONDecodeError):
        replay.read_results(tmp_path)


def test_jpeg_helpers_agree_with_pillow_for_staging(tmp_path):
    p = tmp_path / "x.jpg"
    save_jpeg(p, size=(1024, 768))
    assert "larger" in detector_card.board_decodable(p)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    (tmp_path / "y.jpg").write_bytes(buf.getvalue())
    assert detector_card.board_decodable(tmp_path / "y.jpg") == "not a JPEG"
