"""Inference Engine + Storage + Resume tests（fake provider，不呼叫 API）。"""

from __future__ import annotations

import re

import pytest

from orqa.dataset import load_dataset
from orqa.engine import InferenceAborted, InferenceEngine
from orqa.llm.base import FatalLLMError
from orqa.skill import load_skill
from orqa.storage import (INFERENCE_ERRORS, INVALID_ANSWERS, RAW_PREDICTIONS, RETRIEVALS,
                          ExperimentStorage, append_jsonl, index_records, read_jsonl)
from tests.fakes import FakeProvider, FakeRAG


def correct_letter_by_option(prompt):
    """answer：根據題目文字找出 target（fixture 中 target = index % 4），偶數題答對、奇數題答錯/invalid。"""
    idx = int(re.search(r"Context (\d+)", prompt.user).group(1))
    if idx == 1:
        return "I think the decision is about capacity."   # invalid
    letter = "ABCD"[idx % 4]
    return letter if idx % 2 == 0 else "ABCD"[(idx + 1) % 4]


@pytest.mark.parametrize("mode", ["basic", "skill", "rag", "skill_rag"])
def test_all_modes_share_engine(mode, dataset_file, skill_zip, tmp_path):
    ds = load_dataset(dataset_file)
    skill = load_skill(skill_zip) if "skill" in mode else None
    rag = FakeRAG() if "rag" in mode else None
    provider = FakeProvider(correct_letter_by_option)
    storage = ExperimentStorage(tmp_path / "exp")

    summary = InferenceEngine(provider, mode, skill=skill, rag=rag, experiment_id="e1").run(ds, storage)
    assert summary.written == 6 and summary.invalid == 1 and summary.errors == 0

    raw = storage.raw_predictions()
    assert sorted(raw) == list(range(6))
    r1 = raw[1]
    assert r1["prediction"] is None and r1["is_valid"] is False and r1["raw_response"].startswith("I think")
    r0 = raw[0]
    for key in ("index", "question_type", "prediction", "raw_response", "model_id", "provider",
                "inference_mode", "input_tokens", "output_tokens"):
        assert key in r0
    assert r0["inference_mode"] == mode and r0["prediction"] == 0 and r0["parse_method"] == "strict"

    inv = read_jsonl(storage.path(INVALID_ANSWERS)).records
    assert [r["index"] for r in inv] == [1] and inv[0]["prediction"] is None

    p0 = provider.prompts[0]
    if skill:
        assert skill.instruction in p0.system and skill.instruction not in p0.user
    if rag:
        assert "KNOWLEDGE-0-0" in p0.user and "KNOWLEDGE-0-0" not in p0.system
        assert r0["retrieved_chunk_ids"] == ["doc::0000", "doc::0001"]
        rets = index_records(read_jsonl(storage.path(RETRIEVALS)).records)
        assert rets[0]["chunks"][0]["content"] == "KNOWLEDGE-0-0" and rets[0]["chunks"][0]["dense_rank"] == 1
    else:
        assert not storage.exists(RETRIEVALS)


def test_same_prompt_for_every_provider(dataset_file, skill_zip, tmp_path):
    ds = load_dataset(dataset_file).head(2)
    skill = load_skill(skill_zip)
    a, b = FakeProvider("A", model_id="claude"), FakeProvider("B", model_id="qwen", batch_size=2)
    InferenceEngine(a, "skill_rag", skill=skill, rag=FakeRAG()).run(ds, ExperimentStorage(tmp_path / "a"))
    InferenceEngine(b, "skill_rag", skill=skill, rag=FakeRAG()).run(ds, ExperimentStorage(tmp_path / "b"))
    assert [(p.system, p.user) for p in a.prompts] == [(p.system, p.user) for p in b.prompts]
    assert b.batch_calls == 1  # batch_size=2 → 一次 batch


def test_resume_skips_completed_by_index(dataset_file, tmp_path):
    ds = load_dataset(dataset_file)
    storage = ExperimentStorage(tmp_path / "exp")
    # 先只跑 index 0,2,4（非連續），模擬亂序完成
    subset = [q for q in ds if q.index % 2 == 0]
    InferenceEngine(FakeProvider("A"), "basic").run(subset, storage)

    p = FakeProvider("B")
    summary = InferenceEngine(p, "basic").run(ds, storage)
    assert summary.skipped_completed == 3 and summary.written == 3
    assert len(p.prompts) == 3
    lines = read_jsonl(storage.path(RAW_PREDICTIONS)).records
    assert len(lines) == 6 and len({r["index"] for r in lines}) == 6   # 沒有重複紀錄
    raw = storage.raw_predictions()
    assert raw[0]["raw_response"] == "A" and raw[1]["raw_response"] == "B"

    again = InferenceEngine(FakeProvider("C"), "basic").run(ds, storage)
    assert again.written == 0 and again.skipped_completed == 6


def test_errors_are_logged_and_retried_on_resume(dataset_file, tmp_path):
    ds = load_dataset(dataset_file)
    storage = ExperimentStorage(tmp_path / "exp")
    flaky = FakeProvider("A", fail_on=lambda p: "Context 3" in p.user)
    summary = InferenceEngine(flaky, "basic").run(ds, storage)
    assert summary.written == 5 and summary.errors == 1 and summary.error_indices == [3]
    assert 3 not in storage.completed_indices()                     # 失敗題目不寫入 raw
    err = read_jsonl(storage.path(INFERENCE_ERRORS)).records
    assert err[0]["indices"] == [3] and err[0]["error_type"] == "RuntimeError"

    summary2 = InferenceEngine(FakeProvider("D"), "basic").run(ds, storage)
    assert summary2.written == 1 and storage.raw_predictions()[3]["raw_response"] == "D"


def test_abort_after_consecutive_errors(dataset_file, tmp_path):
    storage = ExperimentStorage(tmp_path / "exp")
    always = FakeProvider("A", fail_on=lambda p: True)
    with pytest.raises(InferenceAborted):
        InferenceEngine(always, "basic").run(load_dataset(dataset_file), storage, max_consecutive_errors=2)
    assert len(always.prompts) == 2


def test_fatal_error_propagates(dataset_file, tmp_path):
    class Fatal(FakeProvider):
        def _generate(self, prompt, *, json_schema=None):
            raise FatalLLMError("bad api key")
    with pytest.raises(FatalLLMError):
        InferenceEngine(Fatal(), "basic").run(load_dataset(dataset_file), ExperimentStorage(tmp_path / "e"))


def test_truncated_last_line_is_tolerated(tmp_path):
    path = tmp_path / "x.jsonl"
    append_jsonl(path, {"index": 0, "v": 1})
    with path.open("ab") as f:
        f.write(b'{"index": 1, "v"')                                   # 模擬寫入中斷
    with pytest.warns(UserWarning, match="corrupt"):
        assert [r["index"] for r in read_jsonl(path).records] == [0]
    append_jsonl(path, {"index": 1, "v": 2})                          # 新紀錄從新的一行開始
    with pytest.warns(UserWarning):
        result = read_jsonl(path)
    assert [r["index"] for r in result.records] == [0, 1] and result.corrupt_lines == [2]


def test_raw_file_is_append_only(tmp_path):
    storage = ExperimentStorage(tmp_path)
    completed = set()
    storage.append_raw_prediction({"index": 0}, completed=completed)
    with pytest.raises(RuntimeError):
        storage.append_raw_prediction({"index": 0}, completed=completed)
    with pytest.raises(RuntimeError):
        storage.write_derived_jsonl(RAW_PREDICTIONS, [])
    with pytest.raises(RuntimeError):
        storage.append(RAW_PREDICTIONS, {"index": 1})


def test_index_records_keeps_first_for_immutable(tmp_path):
    with pytest.warns(UserWarning, match="duplicate"):
        d = index_records([{"index": 1, "v": "a"}, {"index": 1, "v": "b"}], keep="first")
    assert d[1]["v"] == "a"
    assert index_records([{"index": 1, "v": "a"}, {"index": 1, "v": "b"}], keep="last")[1]["v"] == "b"
