import numpy as np

from tinyseek.eval.threshold import best, table


def test_table_counts_and_f1():
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    is_tp = np.array([True, False, True, False])
    rows = {r["threshold"]: r for r in table(scores, is_tp, num_gt=4, thresholds=(0.1, 0.5, 0.85))}
    assert rows[0.1]["detections"] == 4 and rows[0.1]["true_positives"] == 2
    assert rows[0.5]["precision"] == 0.5 and rows[0.5]["recall"] == 0.25
    assert rows[0.85]["precision"] == 1.0 and rows[0.85]["recall"] == 0.25
    assert abs(rows[0.1]["f1"] - 0.5) < 1e-9


def test_best_prefers_higher_threshold_on_ties():
    rows = [{"threshold": 0.3, "f1": 0.5}, {"threshold": 0.6, "f1": 0.5}, {"threshold": 0.9, "f1": 0.2}]
    assert best(rows)["threshold"] == 0.6


def test_no_detections_is_not_an_error():
    rows = table(np.array([]), np.array([], dtype=bool), num_gt=3, thresholds=(0.5,))
    assert rows[0]["recall"] == 0.0 and rows[0]["f1"] == 0.0
