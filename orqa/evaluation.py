"""
orqa/evaluation.py

兩套獨立的 Evaluation：
- Raw           : predictions_raw.jsonl            → evaluation_raw.json / wrong_predictions_raw.json
- Post-analysis : predictions_post_analysis.jsonl  → evaluation_post_analysis.json / wrong_predictions_post_analysis.json

規則
- 一律以 dataset 的題目為準、依 index 對應 prediction（不假設順序或筆數一致）。
- Invalid（prediction = null）與 Missing（沒有紀錄，例如推論失敗）都算錯誤，分開計數。
- accuracy = correct / total_questions。
- 所有比例遇到分母為 0 時回傳 null。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from orqa.dataset import ORQADataset, VALID_QUESTION_TYPES
from orqa.parser import index_to_answer


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _type_sort_key(qtype: str) -> int:
    try:
        return int(qtype[1:])
    except ValueError:
        return 10**6


def evaluate(dataset: ORQADataset, predictions: dict[int, dict[str, Any]], *, kind: str,
             raw_responses: Optional[dict[int, dict]] = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """回傳 (evaluation dict, wrong prediction records)。"""
    raw_responses = raw_responses or predictions
    per_type: dict[str, dict[str, Any]] = {}
    wrong: list[dict[str, Any]] = []
    correct = invalid = missing = 0

    for q in dataset:
        stats = per_type.setdefault(q.question_type, {
            "total": 0, "correct": 0, "incorrect": 0, "invalid": 0, "missing": 0,
            "_pred": Counter(), "_gold": Counter(),
        })
        stats["total"] += 1
        rec = predictions.get(q.index)
        pred = rec.get("prediction") if rec else None

        if rec is not None and pred == q.target_answer:
            correct += 1
            stats["correct"] += 1
            continue

        stats["incorrect"] += 1
        if rec is None:
            missing += 1
            stats["missing"] += 1
            label = "missing"
        elif pred is None:
            invalid += 1
            stats["invalid"] += 1
            label = "invalid"
        else:
            label = index_to_answer(pred)
        stats["_pred"][label] += 1
        stats["_gold"][index_to_answer(q.target_answer)] += 1

        raw = raw_responses.get(q.index) or {}
        wrong.append({
            "index": q.index,
            "question_type": q.question_type,
            "prediction": pred,
            "prediction_letter": index_to_answer(pred),
            "prediction_status": label if label in ("invalid", "missing") else "wrong",
            "prediction_source": (rec or {}).get("prediction_source", "raw" if rec else None),
            "target_answer": q.target_answer,
            "target_answer_letter": index_to_answer(q.target_answer),
            "ground_truth": q.target_answer,              # 規格欄位名稱，與 target_answer 相同
            "ground_truth_letter": index_to_answer(q.target_answer),
            "question": q.question,
            "context": q.context,
            "options": list(q.options),
            "raw_response": raw.get("raw_response"),
        })

    total = len(dataset)
    total_wrong = total - correct
    per_type_out = {}
    for qtype in sorted(per_type, key=_type_sort_key):
        s = per_type[qtype]
        pred_c, gold_c = s.pop("_pred"), s.pop("_gold")
        per_type_out[qtype] = {
            **s,
            "accuracy": safe_div(s["correct"], s["total"]),
            "wrong_distribution": {
                "share_of_all_wrong": safe_div(s["incorrect"], total_wrong),
                "by_predicted": {k: pred_c.get(k, 0) for k in ("A", "B", "C", "D", "invalid", "missing")},
                "by_ground_truth": {k: gold_c.get(k, 0) for k in ("A", "B", "C", "D")},
            },
        }

    result = {
        "kind": kind,
        "total_questions": total,
        "answered": total - missing,
        "accuracy": safe_div(correct, total),
        "correct_count": correct,
        "incorrect_count": total_wrong,
        "invalid_count": invalid,
        "invalid_rate": safe_div(invalid, total),
        "missing_count": missing,
        "per_question_type": per_type_out,
        "question_types_present": [t for t in VALID_QUESTION_TYPES if t in per_type_out],
    }
    return result, wrong


def raw_extras(raw: dict[int, dict]) -> dict[str, Any]:
    methods = Counter(r.get("parse_method") or "invalid" for r in raw.values())
    tokens_in = sum(r.get("input_tokens") or 0 for r in raw.values())
    tokens_out = sum(r.get("output_tokens") or 0 for r in raw.values())
    stops = Counter(r.get("stop_reason") or "unknown" for r in raw.values())
    return {"parse_method_counts": dict(methods), "stop_reason_counts": dict(stops),
            "total_input_tokens": tokens_in, "total_output_tokens": tokens_out}


def post_analysis_metrics(dataset: ORQADataset, raw_eval: dict, post_eval: dict, raw: dict[int, dict],
                          post: dict[int, dict], checked_invalid: int) -> dict[str, Any]:
    original_invalid = sum(1 for r in raw.values() if r.get("prediction") is None)
    statuses = Counter(p.get("analysis_status") for p in post.values()
                       if p.get("raw_prediction") is None)
    resolved = statuses.get("resolved", 0)
    total = raw_eval["total_questions"]
    improvement = (None if raw_eval["accuracy"] is None or post_eval["accuracy"] is None
                   else post_eval["accuracy"] - raw_eval["accuracy"])
    return {
        "original_invalid": original_invalid,
        "resolved_invalid": resolved,
        "unresolved_invalid": statuses.get("unresolved", 0),
        "failed_analysis": statuses.get("failed", 0),
        "not_analyzed": statuses.get("not_analyzed", 0),
        "checked_invalid": checked_invalid,
        "haiku_resolution_rate": safe_div(resolved, original_invalid),
        "checked_invalid_rate": safe_div(checked_invalid, total),
        "checked_invalid_ratio": safe_div(checked_invalid, original_invalid),
        "raw_accuracy": raw_eval["accuracy"],
        "post_analysis_accuracy": post_eval["accuracy"],
        "accuracy_improvement": improvement,
        # Haiku 修復後的答案中有幾題是正確答案（僅供參考：Haiku 只還原原始模型的選擇）
        "haiku_resolved_correct": sum(
            1 for idx, p in post.items()
            if p.get("prediction_source") == "haiku" and dataset.has(idx)
            and p.get("prediction") == dataset.get(idx).target_answer),
    }


def run_evaluation(storage, dataset: ORQADataset) -> dict[str, Any]:
    """產生 evaluation_raw / evaluation_post_analysis / wrong_predictions_* 四個檔案，回傳摘要。"""
    from orqa.analysis import write_post_analysis_files
    from orqa.storage import (CHECKED_INVALID, EVALUATION_POST, EVALUATION_RAW, POST_PREDICTIONS,
                              WRONG_POST, WRONG_RAW, index_records, read_jsonl)

    raw = storage.raw_predictions()
    if not storage.exists(POST_PREDICTIONS):
        write_post_analysis_files(storage, dataset)
    post = index_records(read_jsonl(storage.path(POST_PREDICTIONS)).records, name=POST_PREDICTIONS)
    checked = read_jsonl(storage.path(CHECKED_INVALID)).records

    extra_raw = sorted(set(raw) - set(dataset.indices))
    raw_eval, raw_wrong = evaluate(dataset, raw, kind="raw")
    raw_eval.update(raw_extras(raw))
    raw_eval["predictions_outside_evaluated_dataset"] = len(extra_raw)

    post_eval, post_wrong = evaluate(dataset, post, kind="post_analysis", raw_responses=raw)
    post_eval["remaining_invalid_count"] = post_eval["invalid_count"]
    post_eval["remaining_invalid_rate"] = post_eval["invalid_rate"]
    post_eval["prediction_source_counts"] = dict(Counter(p.get("prediction_source") for p in post.values()))
    in_scope_checked = sum(1 for c in checked if dataset.has(int(c["index"])))
    post_eval["post_analysis_metrics"] = post_analysis_metrics(
        dataset, raw_eval, post_eval,
        {i: r for i, r in raw.items() if dataset.has(i)},
        {i: p for i, p in post.items() if dataset.has(i)},
        in_scope_checked)

    storage.write_json(EVALUATION_RAW, raw_eval)
    storage.write_json(EVALUATION_POST, post_eval)
    storage.write_json(WRONG_RAW, raw_wrong)
    storage.write_json(WRONG_POST, post_wrong)
    return {"raw": raw_eval, "post": post_eval}
