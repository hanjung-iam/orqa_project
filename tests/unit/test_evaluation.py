"""Evaluation unit tests."""

from __future__ import annotations

import pytest

from orqa.dataset import load_dataset
from orqa.evaluation import evaluate, post_analysis_metrics, safe_div


def test_safe_div():
    assert safe_div(1, 0) is None and safe_div(1, 4) == 0.25


def test_alignment_by_index_not_order(dataset_file):
    ds = load_dataset(dataset_file)  # targets: i % 4
    preds = {5: {"index": 5, "prediction": 1}, 0: {"index": 0, "prediction": 0},
             3: {"index": 3, "prediction": None}, 99: {"index": 99, "prediction": 0}}
    result, wrong = evaluate(ds, preds, kind="raw")
    assert result["correct_count"] == 2 and result["invalid_count"] == 1
    assert result["missing_count"] == 3 and result["incorrect_count"] == 4
    assert result["accuracy"] == pytest.approx(2 / 6) and result["invalid_rate"] == pytest.approx(1 / 6)
    assert sorted(w["index"] for w in wrong) == [1, 2, 3, 4]
    statuses = {w["index"]: w["prediction_status"] for w in wrong}
    assert statuses[3] == "invalid" and statuses[1] == "missing"


def test_per_type_stats_and_distribution(dataset_file):
    ds = load_dataset(dataset_file)  # types: Q1,Q2,Q3,Q10,Q2,Q11
    preds = {i: {"index": i, "prediction": 3} for i in range(6)}   # 只有 index 3 正確
    result, _ = evaluate(ds, preds, kind="raw")
    pt = result["per_question_type"]
    assert list(pt) == ["Q1", "Q2", "Q3", "Q10", "Q11"]              # Q 編號排序，不是字串排序
    assert pt["Q2"]["total"] == 2 and pt["Q2"]["accuracy"] == 0.0
    assert pt["Q10"]["accuracy"] == 1.0
    assert pt["Q2"]["wrong_distribution"]["by_predicted"]["D"] == 2
    assert pt["Q2"]["wrong_distribution"]["by_ground_truth"] == {"A": 1, "B": 1, "C": 0, "D": 0}
    assert sum(t["wrong_distribution"]["share_of_all_wrong"] for t in pt.values() if t["incorrect"]) == pytest.approx(1)


def test_metrics_without_invalid(dataset_file):
    ds = load_dataset(dataset_file)
    preds = {i: {"index": i, "prediction": i % 4, "raw_prediction": i % 4, "prediction_source": "raw"}
             for i in range(6)}
    raw_eval, _ = evaluate(ds, preds, kind="raw")
    post_eval, _ = evaluate(ds, preds, kind="post")
    m = post_analysis_metrics(ds, raw_eval, post_eval, preds, preds, 0)
    assert m["original_invalid"] == 0 and m["haiku_resolution_rate"] is None
    assert m["checked_invalid_ratio"] is None and m["checked_invalid_rate"] == 0.0
    assert m["accuracy_improvement"] == 0.0
