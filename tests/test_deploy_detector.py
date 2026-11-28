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
                    str(DET / "src" / "kv.cpp"), str(DET / "host_test" / "logic_cli.cpp"), "-o", str(out)], check=True)
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


# --- review fixes (2026-11-26) -----------------------------------------------------------------

def test_uncommented_footer_line_wins_over_the_staged_value(cli, tmp_path):
    """#1: '#   wifi=sta' with only the '#' deleted leaves leading spaces; the key is still
    'wifi' and, coming later, overrides the staged wifi=off."""
    s, warnings = settings(cli, tmp_path, "prompt=backpack\r\nwifi=off\r\n   wifi=sta\r\n   wifi_ssid=net\r\n"
                                          "\twifi_password = secret123 \r\n")
    assert warnings == []
    assert (s["wifi"], s["wifi_ssid"]) == ("sta", "net")
    s, _ = settings(cli, tmp_path, "led=off\nled = onboard\n  # led=external\n")
    assert s["led"] == "onboard"  # later line wins; an indented '#' line is a comment


def test_with_prompt_matches_spaced_keys_and_drops_duplicates(cli, tmp_path):
    """#1: 'prompt = cup' is the prompt line; a second prompt line must not survive."""
    cfg = tmp_path / "c.txt"
    cfg.write_bytes(b"prompt = cup\r\nsource=camera\r\n  prompt=mug\n# prompt=old\n")
    out = subprocess.run([str(cli), "with_prompt", str(cfg), "backpack"], capture_output=True, check=True).stdout
    assert out == b"prompt=backpack\nsource=camera\r\n# prompt=old\n"
    cfg.write_bytes(out)
    s, warnings = settings(cli, tmp_path, out.decode())
    assert s["prompt"] == "backpack" and warnings == []


def test_bom_is_ignored(cli, tmp_path):
    """#7: a UTF-8 BOM (Windows Notepad) before the first key."""
    s, warnings = settings(cli, tmp_path, "﻿prompt=cup\nsource=replay\n")
    assert warnings == [] and (s["prompt"], s["source"]) == ("cup", "replay")
    cfg = tmp_path / "bom.txt"
    cfg.write_bytes(b"\xef\xbb\xbfprompt=cup\nsource=replay\n")
    out = subprocess.run([str(cli), "with_prompt", str(cfg), "mug"], capture_output=True, check=True).stdout
    assert out == b"\xef\xbb\xbfprompt=mug\nsource=replay\n"
    assert read_kv(cfg) == {"prompt": "cup", "source": "replay"}  # the PC reader agrees


def test_prompt_save_never_drops_an_unreadable_config(cli, tmp_path):
    """#2: config.txt exists but cannot be read -> skip the save; missing -> prompt only."""
    cfg = tmp_path / "c.txt"
    cfg.write_text("wifi=sta\nprompt=cup\n")
    assert run(cli, "save", "1", "0", str(cfg), "mug") == ["SKIP"]
    assert run(cli, "save", "1", "1", str(cfg), "mug") == ["wifi=sta", "prompt=mug"]
    assert run(cli, "save", "0", "0", str(cfg), "mug") == ["prompt=mug"]
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert run(cli, "save", "1", "1", str(empty), "mug") == ["prompt=mug"]


@pytest.mark.parametrize("cfg,tmp,action", [("1", "0", "none"), ("0", "0", "none"), ("0", "1", "restore"),
                                            ("1", "1", "delete")])
def test_config_recovery_after_an_interrupted_save(cli, cfg, tmp, action):
    """#2: write config.tmp, remove config.txt, rename. Only the copy -> restore it; both ->
    config.txt is intact and the copy is stale."""
    assert run(cli, "recover", cfg, tmp) == [action]


@pytest.mark.parametrize("pending,current,replay,action", [
    ("cup", "backpack", "0", "switch"), ("cup", "backpack", "1", "refuse"), ("backpack", "backpack", "0", "none"),
    ("", "backpack", "1", "none")])
def test_prompt_switch_is_refused_during_replay(cli, pending, current, replay, action):
    """#5: a request queued before the replay began must not switch the model mid-replay."""
    assert run(cli, "action", pending, current, replay) == [action]


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


def test_stage_w16a16_run_records_int16_io(tmp_path, cli):
    run16 = fake_run(tmp_path / "runs" / "w16", meta_extra={"quant": "w16a16"})
    (run16 / "model.info").write_text(INFO.replace("INT8, 1x192x192x3], exponents: [-6]", "INT16, 1x192x192x3], exponents: [-13]")
                                      .replace("%641[INT8, 1x24x24x5], exponents: [-4]", "%641[INT16, 1x24x24x5], exponents: [-12]"))
    out = tmp_path / "card"
    detector_card.stage([(None, run16)], out, None, {}, None, {}, None, None, 0, None)
    side = out / "sdcard" / "tinyseek" / "models" / "backpack.txt"
    kv = read_kv(side)
    assert (kv["input_dtype"], kv["output_dtype"], kv["input_exponent"], kv["output_exponent"], kv["quant"]) == \
        ("INT16", "INT16", "-13", "-12", "w16a16")
    meta = dict(line.split("=", 1) for line in run(cli, "meta", str(side)))
    assert (meta["input_dtype"], meta["output_dtype"], meta["input_exponent"]) == ("INT16", "INT16", "-13")
    (run16 / "model.info").write_text(INFO.replace("[INT8, 1x192x192x3]", "[FP32, 1x192x192x3]"))
    with pytest.raises(ValueError):
        detector_card.stage([(None, run16)], out, None, {}, None, {}, None, None, 0, None)


# --- rectangular model inputs (2026-11-28) -------------------------------------------------------

RECT_INFO = INFO.replace("1x192x192x3", "1x192x256x3").replace("1x24x24x5", "1x24x32x5")  # NHWC: 192 rows, 256 columns
SQUARE_SIDECAR_KEYS = ["prompt", "run", "input_size", "stride", "num_classes", "mean", "std", "pad_value",
                       "score_threshold", "max_detections", "detect_threshold", "detect_threshold_source",
                       "input_exponent", "output_exponent", "input_dtype", "output_dtype", "quant", "espdl_bytes",
                       "espdl_sha256"]


def rect_run(path: Path) -> Path:
    """fake_run with a 256x192 (width x height) model, as tinyseek.train.export writes it."""
    run_dir = fake_run(path)
    meta = json.loads((run_dir / "model_meta.json").read_text())
    del meta["input_size"]
    meta.update(input_width=256, input_height=192)
    (run_dir / "model_meta.json").write_text(json.dumps(meta))
    (run_dir / "model.info").write_text(RECT_INFO)
    return run_dir


def cpp_meta(cli: Path, path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in run(cli, "meta", str(path)))


def test_square_sidecar_is_what_it_was(tmp_path, cli):
    """A square model's sidecar keeps its keys, in order, with input_size and no width/height."""
    out = tmp_path / "card"
    m = detector_card.stage([(None, fake_run(tmp_path / "r"))], out, None, {}, None, {}, None, None, 0, None)
    side = out / "sdcard" / "tinyseek" / "models" / "backpack.txt"
    lines = side.read_text().splitlines()
    assert [line.split("=", 1)[0] for line in lines] == SQUARE_SIDECAR_KEYS
    assert "input_size=192" in lines
    assert m["models"][0]["input_size"] == 192 and "input_width" not in m["models"][0]
    meta = cpp_meta(cli, side)
    assert (meta["input_size"], meta["input_width"], meta["input_height"]) == ("192", "192", "192")


def test_rectangular_sidecar_has_width_and_height(tmp_path, cli):
    out = tmp_path / "card"
    m = detector_card.stage([(None, rect_run(tmp_path / "r"))], out, None, {}, None, {}, None, None, 0, None)
    side = out / "sdcard" / "tinyseek" / "models" / "backpack.txt"
    kv = read_kv(side)
    assert "input_size" not in kv and (kv["input_width"], kv["input_height"]) == ("256", "192")
    keys = [line.split("=", 1)[0] for line in side.read_text().splitlines()]
    assert keys == ["prompt", "run", "input_width", "input_height", *SQUARE_SIDECAR_KEYS[3:]]
    staged = m["models"][0]
    assert (staged["input_width"], staged["input_height"]) == (256, 192) and "input_size" not in staged
    assert (kv["input_exponent"], kv["output_exponent"]) == ("-6", "-4")
    # The firmware's parser reads the new keys; a rectangular size is not reported as input_size.
    meta = cpp_meta(cli, side)
    assert "input_size" not in meta and (meta["input_width"], meta["input_height"]) == ("256", "192")


def test_staging_checks_the_rectangular_graph_shape(tmp_path):
    r = rect_run(tmp_path / "r")
    args = ([(None, r)], tmp_path / "card", None, {}, None, {}, None, None, 0, None)
    (r / "model.info").write_text(RECT_INFO.replace("1x192x256x3", "1x256x192x3"))  # width and height swapped
    with pytest.raises(SystemExit, match="256x192"):
        detector_card.stage(*args)
    (r / "model.info").write_text(RECT_INFO.replace("1x24x32x5", "1x32x24x5"))  # grid swapped
    with pytest.raises(SystemExit, match="grid"):
        detector_card.stage(*args)
    (r / "model.info").write_text(INFO)  # a square graph for rectangular metadata
    with pytest.raises(SystemExit):
        detector_card.stage(*args)
    (r / "model.info").write_text(RECT_INFO)
    detector_card.stage(*args)
    # No size at all in model_meta.json is an error, not a guess.
    meta = json.loads((r / "model_meta.json").read_text())
    del meta["input_width"]
    (r / "model_meta.json").write_text(json.dumps(meta))
    with pytest.raises(KeyError):
        detector_card.stage(*args)


@pytest.mark.parametrize("text,size,width,height", [
    ("input_size=128\n", "128", "128", "128"),                              # a card made before rectangular inputs
    ("input_width=256\ninput_height=192\n", None, "256", "192"),          # a rectangular model
    ("input_width=288\ninput_height=224\ninput_size=192\n", None, "288", "224"),  # both: width/height win
    ("input_width=256\n", "0", "0", "0"),                                   # half a size is no size
    ("prompt=cup\n", "0", "0", "0"),                                        # not given: nothing to compare
    ("input_width=192\ninput_height=192\n", "192", "192", "192"),         # square, written the new way
])
def test_firmware_reads_old_and_new_size_keys(cli, tmp_path, text, size, width, height):
    path = tmp_path / "m.txt"
    path.write_text(text)
    meta = cpp_meta(cli, path)
    assert (meta.get("input_size"), meta["input_width"], meta["input_height"]) == (size, width, height)


def test_firmware_reports_the_model_shape(cli):
    """Square models keep "input"/"grid" and input_size; rectangular ones get their own keys."""
    assert run(cli, "shape", "192", "192", "24", "24") == ['"input":192,"grid":24', "input_size=192"]
    assert run(cli, "shape", "256", "192", "32", "24") == [
        '"input_w":256,"input_h":192,"grid_w":32,"grid_h":24', "input_width=256", "input_height=192"]
    assert json.loads("{" + run(cli, "shape", "288", "224", "36", "28")[0] + "}") == {
        "input_w": 288, "input_h": 224, "grid_w": 36, "grid_h": 28}


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
