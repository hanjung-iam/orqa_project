"""Post-analysis tests：raw 不可變、冪等、resolved/unresolved/failed、依 index 合併。"""

from __future__ import annotations

import hashlib
import json

import pytest

from orqa.analysis import (ANALYSIS_SCHEMA, InvalidAnswerAnalyzer, build_analysis_prompt,
                           merge_predictions, run_post_analysis)
from orqa.dataset import load_dataset
from orqa.engine import InferenceEngine
from orqa.llm.base import FatalLLMError, LLMResponse
from orqa.storage import (CHECKED_INVALID, HAIKU_ANALYSIS, POST_PREDICTIONS, RAW_PREDICTIONS,
                          ExperimentStorage, read_jsonl)
from tests.fakes import FakeProvider

RESPONSES = {
    0: "A",                                                   # valid
    1: "I believe the right choice is the capacity limit, i.e. option C overall.",   # Haiku → C
    2: "It could be A or B, hard to say.",                    # Haiku → UNRESOLVED
    3: "Analysis of the objective function and",              # Haiku API failure
    4: "",                                                    # empty → unresolved without API
    5: "D",
}


def _answer(prompt):
    idx = int(prompt.user.split("Context ")[1].split("\n")[0])
    return RESPONSES[idx]


class FakeHaiku(FakeProvider):
    def _generate(self, prompt, *, json_schema=None):
        assert json_schema == ANALYSIS_SCHEMA
        self.prompts.append(prompt)
        resp = prompt.user.split("<response>\n")[1]
        if "capacity limit" in resp:
            out = {"evidence": "option C overall", "selected_option": "C"}
        elif "A or B" in resp:
            out = {"evidence": "two options", "selected_option": "UNRESOLVED"}
        else:
            raise RuntimeError("simulated API error")
        text = json.dumps(out)
        return LLMResponse(text=text, model_id="claude-haiku-4-5-20251001", provider="anthropic",
                           input_tokens=100, output_tokens=20, stop_reason="end_turn")


@pytest.fixture
def run(dataset_file, tmp_path):
    ds = load_dataset(dataset_file)
    storage = ExperimentStorage(tmp_path / "exp")
    InferenceEngine(FakeProvider(_answer), "basic").run(ds, storage)
    return ds, storage


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_post_analysis_end_to_end(run):
    ds, storage = run
    raw_hash = sha(storage.path(RAW_PREDICTIONS))
    haiku = FakeHaiku()
    summary = run_post_analysis(storage, ds, InvalidAnswerAnalyzer(haiku, "claude-haiku-4-5-20251001"))

    assert sha(storage.path(RAW_PREDICTIONS)) == raw_hash                  # raw 完全不變
    assert storage.raw_predictions()[1]["prediction"] is None
    assert summary.original_invalid == 4
    assert (summary.resolved, summary.unresolved, summary.failed) == (1, 2, 1)
    assert len(haiku.prompts) == 3                                         # 空回答不呼叫 API

    analyses = storage.latest(HAIKU_ANALYSIS)
    a1 = analyses[1]
    for key in ("index", "original_prediction", "recovered_prediction", "analysis_status",
                "analysis_response", "model_id", "input_tokens", "output_tokens", "timestamp"):
        assert key in a1
    assert a1["analysis_status"] == "resolved" and a1["recovered_prediction"] == 2
    assert analyses[2]["analysis_status"] == "unresolved"
    assert analyses[3]["analysis_status"] == "failed" and "simulated" in analyses[3]["failure_reason"]
    assert analyses[4]["analysis_status"] == "unresolved" and "empty_response" in analyses[4]["failure_reason"]

    post = {r["index"]: r for r in read_jsonl(storage.path(POST_PREDICTIONS)).records}
    assert post[0]["prediction"] == 0 and post[0]["prediction_source"] == "raw"
    assert post[1]["prediction"] == 2 and post[1]["prediction_source"] == "haiku"
    assert post[2]["prediction"] is None and post[3]["prediction"] is None

    checked = {r["index"]: r for r in read_jsonl(storage.path(CHECKED_INVALID)).records}
    assert sorted(checked) == [2, 3, 4]
    for key in ("index", "original_prediction", "raw_response", "analysis_status", "failure_reason"):
        assert key in checked[3]


def test_idempotent_and_retry_failed(run):
    ds, storage = run
    run_post_analysis(storage, ds, InvalidAnswerAnalyzer(FakeHaiku(), "h"))
    second = FakeHaiku()
    summary = run_post_analysis(storage, ds, InvalidAnswerAnalyzer(second, "h"))
    assert summary.already_done == 3 and summary.analyzed_now == 1        # 只重試 failed 的 index 3
    assert len(second.prompts) == 1 and "objective function" in second.prompts[0].user

    third = FakeHaiku()
    run_post_analysis(storage, ds, InvalidAnswerAnalyzer(third, "h"), reanalyze=True)
    assert len(third.prompts) == 3                                         # 明確要求才全部重做


def test_disabled_post_analysis_marks_not_analyzed(run):
    ds, storage = run
    summary = run_post_analysis(storage, ds, None)
    assert summary.analyzed_now == 0 and not storage.exists(HAIKU_ANALYSIS)
    checked = read_jsonl(storage.path(CHECKED_INVALID)).records
    assert {c["analysis_status"] for c in checked} == {"not_analyzed"} and len(checked) == 4


def test_fatal_error_stops(run):
    ds, storage = run

    class Fatal(FakeHaiku):
        def _generate(self, prompt, *, json_schema=None):
            raise FatalLLMError("auth")
    with pytest.raises(FatalLLMError):
        run_post_analysis(storage, ds, InvalidAnswerAnalyzer(Fatal(), "h"))


def test_merge_uses_index_not_order():
    raw = {5: {"index": 5, "prediction": None, "raw_response": "x", "question_type": "Q1"},
           2: {"index": 2, "prediction": 1, "raw_response": "B", "question_type": "Q2"}}
    analyses = {5: {"index": 5, "analysis_status": "resolved", "recovered_prediction": 3}}
    post, checked = merge_predictions(raw, analyses)
    assert [(p["index"], p["prediction"]) for p in post] == [(2, 1), (5, 3)] and checked == []


def test_analysis_prompt_does_not_include_question(dataset_file):
    q = load_dataset(dataset_file).get(2)
    p = build_analysis_prompt(q, "some reasoning")
    assert "opt2c" in p.user and "Context 2" not in p.user and "Question 2" not in p.user
    assert "NOT to answer the question" in p.system
