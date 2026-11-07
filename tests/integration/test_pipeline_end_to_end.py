"""
End-to-end pipeline tests through main.py（fake LLM / fake Haiku，不呼叫 API）：
新實驗 → 中斷 → resume → post-analysis → evaluation；實驗隔離與舊結果保護。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

import main as runner
import orqa.llm as llm_module
from orqa.experiment import ExperimentError
from orqa.storage import (CHECKED_INVALID, EVALUATION_POST, EVALUATION_RAW, EXPERIMENT, HAIKU_ANALYSIS,
                          INVALID_ANSWERS, POST_PREDICTIONS, RAW_PREDICTIONS, WRONG_POST, WRONG_RAW,
                          read_jsonl)
from tests.fakes import FakeProvider
from tests.unit.test_post_analysis import FakeHaiku


def _answer(prompt):
    i = int(re.search(r"Context (\d+)", prompt.user).group(1))
    # 0,2,4 正確；1 invalid（Haiku 可還原為 C）；3 invalid（Haiku UNRESOLVED）；5 錯誤
    return {0: "A", 1: "I believe the right choice is the capacity limit, i.e. option C overall.",
            2: "The answer is (C).", 3: "It could be A or B, hard to say.", 4: "**A**", 5: "A"}[i]


@pytest.fixture
def setup(tmp_path, dataset_file, monkeypatch):
    out = tmp_path / "results"
    legacy = out / "Basic"
    legacy.mkdir(parents=True)
    (legacy / "basic_predictions.jsonl").write_text('{"index": 0, "prediction": 2}\n', encoding="utf-8")
    cfg = {
        "experiment": {"method": "basic", "output_dir": str(out), "max_consecutive_errors": 3},
        "dataset": {"name": "mini", "path": str(dataset_file)},
        "llm": {"provider": "anthropic", "anthropic": {"model_id": "fake-claude"}},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    state = {"fail": set()}
    monkeypatch.setattr(llm_module, "create_provider",
                        lambda s: FakeProvider(_answer, model_id=s.active.model_id,
                                               fail_on=lambda p: any(f"Context {i}\n" in p.user for i in state["fail"])))
    monkeypatch.setattr(llm_module, "create_post_analysis_provider", lambda s: FakeHaiku())
    return cfg_path, out, legacy, state


def tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def test_full_pipeline_with_interrupt_and_resume(setup):
    cfg_path, out, legacy, state = setup
    legacy_hash = tree_hash(legacy)

    # 1) index 3 一直失敗（模擬 API 錯誤）→ 推論不完整
    state["fail"] = {3}
    assert runner.main(["--config", str(cfg_path), "--experiment-id", "exp_a"]) == 0
    exp_dir = out / "exp_a"
    meta = json.loads((exp_dir / EXPERIMENT).read_text(encoding="utf-8"))
    assert meta["experiment_status"] == "interrupted"
    assert len(read_jsonl(exp_dir / RAW_PREDICTIONS).records) == 5

    for key in ("experiment_id", "timestamp", "dataset_name", "provider", "model_id", "quantization",
                "inference_mode", "skill_enabled", "RAG_enabled", "embedding_model", "similarity_threshold",
                "dense_top_k", "bm25_top_k", "rrf_k", "final_top_k", "generation_parameters", "experiment_status"):
        assert key in meta, key
    assert meta["model_id"] == "fake-claude" and meta["inference_mode"] == "basic"
    assert meta["dataset"]["sha256"] and meta["prompt"]["version"] == "v2"

    # 2) resume：只補 index 3，不重複任何題目
    state["fail"] = set()
    assert runner.main(["--config", str(cfg_path), "--resume", "exp_a"]) == 0
    raw_lines = read_jsonl(exp_dir / RAW_PREDICTIONS).records
    assert sorted(r["index"] for r in raw_lines) == list(range(6))
    meta = json.loads((exp_dir / EXPERIMENT).read_text(encoding="utf-8"))
    assert meta["experiment_status"] == "evaluated"
    assert [r["stage"] for r in meta["runs"]][:2] == ["inference", "post_analysis"]

    for name in (RAW_PREDICTIONS, INVALID_ANSWERS, HAIKU_ANALYSIS, CHECKED_INVALID, POST_PREDICTIONS,
                 EVALUATION_RAW, EVALUATION_POST, WRONG_RAW, WRONG_POST):
        assert (exp_dir / name).exists(), name

    raw_eval = json.loads((exp_dir / EVALUATION_RAW).read_text(encoding="utf-8"))
    post_eval = json.loads((exp_dir / EVALUATION_POST).read_text(encoding="utf-8"))
    # targets: i % 4 → 0:A 1:B 2:C 3:D 4:A 5:B
    assert (raw_eval["correct_count"], raw_eval["invalid_count"]) == (3, 2)
    assert raw_eval["accuracy"] == pytest.approx(3 / 6)
    assert raw_eval["parse_method_counts"] == {"strict": 2, "statement": 1, "wrapped": 1, "invalid": 2}
    # Haiku 把 index 1 還原成 C（gold = B，仍然錯）；index 3 unresolved
    assert (post_eval["correct_count"], post_eval["remaining_invalid_count"]) == (3, 1)
    m = post_eval["post_analysis_metrics"]
    assert m["original_invalid"] == 2 and m["resolved_invalid"] == 1
    assert m["haiku_resolution_rate"] == 0.5 and m["checked_invalid"] == 1
    assert m["checked_invalid_rate"] == pytest.approx(1 / 6) and m["checked_invalid_ratio"] == 0.5
    assert m["accuracy_improvement"] == 0.0

    wrong_raw = json.loads((exp_dir / WRONG_RAW).read_text(encoding="utf-8"))
    assert sorted(w["index"] for w in wrong_raw) == [1, 3, 5]
    w5 = next(w for w in wrong_raw if w["index"] == 5)
    for key in ("index", "question_type", "prediction", "ground_truth", "question", "context", "options", "raw_response"):
        assert key in w5
    assert (w5["prediction_letter"], w5["ground_truth_letter"]) == ("A", "B")
    wrong_post = json.loads((exp_dir / WRONG_POST).read_text(encoding="utf-8"))
    w1 = next(w for w in wrong_post if w["index"] == 1)
    assert w1["prediction_source"] == "haiku" and w1["prediction_letter"] == "C" and w1["raw_response"].startswith("I believe")

    # 3) 舊結果完全沒有被動到
    assert tree_hash(legacy) == legacy_hash


def test_post_analysis_stage_does_not_touch_raw_and_is_idempotent(setup):
    cfg_path, out, _, _ = setup
    runner.main(["--config", str(cfg_path), "--experiment-id", "exp_b"])
    raw = out / "exp_b" / RAW_PREDICTIONS
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    n_analysis = len(read_jsonl(out / "exp_b" / HAIKU_ANALYSIS).records)
    assert runner.main(["--config", str(cfg_path), "--resume", "exp_b", "--stage", "post_analysis"]) == 0
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before
    assert len(read_jsonl(out / "exp_b" / HAIKU_ANALYSIS).records) == n_analysis  # 沒有重複呼叫


def test_new_experiments_never_overwrite(setup):
    cfg_path, out, _, _ = setup
    runner.main(["--config", str(cfg_path), "--experiment-id", "exp_c"])
    with pytest.raises(ExperimentError, match="already exists"):
        runner.main(["--config", str(cfg_path), "--experiment-id", "exp_c"])
    with pytest.raises(ExperimentError, match="already exists"):
        runner.main(["--config", str(cfg_path), "--experiment-id", "Basic"])   # 舊結果目錄


def test_resume_refuses_changed_dataset(setup, dataset_file):
    cfg_path, out, _, _ = setup
    runner.main(["--config", str(cfg_path), "--experiment-id", "exp_d", "--limit", "2"])
    with dataset_file.open("a", encoding="utf-8") as f:
        f.write('{"QUESTION_TYPE":"Q1","CONTEXT":"x","QUESTION":"y","OPTIONS":["a","b","c","d"],"TARGET_ANSWER":0}\n')
    with pytest.raises(ExperimentError, match="dataset"):
        runner.main(["--config", str(cfg_path), "--resume", "exp_d"])


def test_limit_then_extend(setup):
    cfg_path, out, _, _ = setup
    runner.main(["--config", str(cfg_path), "--experiment-id", "exp_e", "--limit", "2"])
    assert len(read_jsonl(out / "exp_e" / RAW_PREDICTIONS).records) == 2
    ev = json.loads((out / "exp_e" / EVALUATION_RAW).read_text(encoding="utf-8"))
    assert ev["total_questions"] == 2
    runner.main(["--config", str(cfg_path), "--resume", "exp_e", "--limit", "6"])
    assert len(read_jsonl(out / "exp_e" / RAW_PREDICTIONS).records) == 6
    ev = json.loads((out / "exp_e" / EVALUATION_RAW).read_text(encoding="utf-8"))
    assert ev["total_questions"] == 6


def test_fatal_error_stops_before_post_analysis(setup, monkeypatch):
    cfg_path, out, _, _ = setup
    from orqa.llm.base import FatalLLMError

    class BadKey(FakeProvider):
        def _generate(self, prompt, *, json_schema=None):
            raise FatalLLMError("401 invalid x-api-key")
    monkeypatch.setattr(llm_module, "create_provider", lambda s: BadKey())
    assert runner.main(["--config", str(cfg_path), "--experiment-id", "exp_f"]) == 1
    meta = json.loads((out / "exp_f" / EXPERIMENT).read_text(encoding="utf-8"))
    assert meta["experiment_status"] == "failed"
    assert not (out / "exp_f" / EVALUATION_RAW).exists() and not (out / "exp_f" / RAW_PREDICTIONS).exists()


def test_dry_run_creates_nothing(setup, capsys):
    cfg_path, out, _, _ = setup
    assert runner.main(["--config", str(cfg_path), "--dry-run"]) == 0
    assert "[SYSTEM]" in capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == ["Basic"]
