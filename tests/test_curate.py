from tinyseek.label.curate import box_id, curate


def test_box_id_is_stable_under_rounding():
    assert box_id("cup", [10.2, 9.8, 20.0, 20.4]) == box_id("cup", [10.4, 10.1, 19.9, 20.0])
    assert box_id("cup", [10, 10, 20, 20]) != box_id("backpack", [10, 10, 20, 20])


def test_curate_applies_verdicts():
    coco = {"images": [{"id": 1, "file_name": "a.jpg"}], "categories": [{"id": 47, "name": "cup"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 47, "bbox": [10.2, 10, 20, 20], "score": 0.15},  # low, good
                {"id": 2, "image_id": 1, "category_id": 47, "bbox": [50, 50, 20, 20], "score": 0.90},    # high, bad
                {"id": 3, "image_id": 1, "category_id": 47, "bbox": [90, 90, 20, 20], "score": 0.40},    # unreviewed
                {"id": 4, "image_id": 1, "category_id": 47, "bbox": [5, 90, 20, 20], "score": 0.10},     # unreviewed
            ]}
    verdicts = {("a.jpg", box_id("cup", [10.4, 9.8, 20, 20])): "good",
                ("a.jpg", box_id("cup", [50, 50, 20, 20])): "bad"}
    out, counts = curate(coco, verdicts, threshold=0.3)
    assert [a["bbox"][0] for a in out["annotations"]] == [10.2, 90]
    assert counts == {"good": 1, "bad": 1, "threshold_kept": 1, "threshold_dropped": 1}
    assert out["annotations"][0]["verified"] is True
    assert out["annotations"][1]["verified"] is False
