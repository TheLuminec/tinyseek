import json
import sys

import pytest

from tinyseek.label import filter as label_filter


def _raw(tmp_path):
    coco = {
        "info": {"min_score": 0.1},
        "images": [{"id": 1, "file_name": "a.jpg", "width": 8, "height": 8}, {"id": 2, "file_name": "b.jpg", "width": 8, "height": 8}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 0.12},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "score": 0.18},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [2, 2, 2, 2], "score": 0.4},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    path = tmp_path / "synth_set_person_raw.json"
    path.write_text(json.dumps(coco))
    return path


def test_filter_writes_threshold_copy(tmp_path, monkeypatch):
    raw = _raw(tmp_path)
    monkeypatch.setattr(sys, "argv", ["filter", "--raw", str(raw), "--threshold", "0.15"])
    label_filter.main()
    out = json.loads((tmp_path / "synth_set_person_t0.15.json").read_text())
    assert [a["id"] for a in out["annotations"]] == [2, 3]
    assert len(out["images"]) == 2  # images without a box stay, as negatives


def test_filter_refuses_threshold_below_raw_floor(tmp_path, monkeypatch):
    raw = _raw(tmp_path)
    monkeypatch.setattr(sys, "argv", ["filter", "--raw", str(raw), "--threshold", "0.05"])
    with pytest.raises(SystemExit):
        label_filter.main()
