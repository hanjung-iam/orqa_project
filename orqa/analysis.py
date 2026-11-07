"""
orqa/analysis.py

Invalid Answer Post-analysis（Claude Haiku）。

目的：根據原始模型的回答（raw_response），判斷「原始模型實際選擇了哪個選項」。
Haiku 不重新作答；回答沒有明確選擇、選了多個、被截斷或含糊時一律回 UNRESOLVED。

- 輸出使用 structured outputs（JSON schema，selected_option ∈ {A,B,C,D,UNRESOLVED}）
- 只送「四個選項 + 原始回答」，不送 Context / Question，降低 Haiku 自行解題的可能
- 結果 append 到 haiku_analysis.jsonl；predictions_raw.jsonl 只讀不寫
- 冪等：已 resolved / unresolved 的 index 不會再呼叫 API（除非 reanalyze=True）；
  failed 的 index 下次執行會重試
- 合併結果（依 index）寫成 predictions_post_analysis.jsonl 與 checked_invalid.jsonl（衍生檔，可重建）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from orqa.dataset import ORQADataset, Question
from orqa.llm.base import BaseLLMProvider, FatalLLMError
from orqa.parser import ANSWER_TO_INDEX
from orqa.prompts import ChatPrompt
from orqa.storage import CHECKED_INVALID, HAIKU_ANALYSIS, POST_PREDICTIONS, ExperimentStorage

ANALYSIS_PROMPT_VERSION = "a1"

ANALYSIS_SYSTEM = (
    "You are reviewing the output of another language model that answered a four-option "
    "multiple-choice question (options A, B, C, D).\n"
    "Your task is NOT to answer the question yourself, and NOT to judge which option is correct.\n"
    "Determine which single option the response itself finally selected.\n"
    "Rules:\n"
    "- Use only what the response says.\n"
    "- If the response clearly commits to exactly one option, either by its letter or by unambiguously "
    "restating that option's text, return that letter.\n"
    "- If the response does not commit to an option, commits to more than one option, is cut off before "
    "a final choice, or is otherwise ambiguous, return UNRESOLVED.\n"
    "- Do not guess.\n"
    "In evidence, quote the short part of the response that shows the final choice, or explain briefly "
    "why it is unresolved."
)

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "selected_option": {"type": "string", "enum": ["A", "B", "C", "D", "UNRESOLVED"]},
    },
    "required": ["evidence", "selected_option"],
    "additionalProperties": False,
}

RESOLVED, UNRESOLVED, FAILED = "resolved", "unresolved", "failed"
FINAL_STATUSES = (RESOLVED, UNRESOLVED)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_analysis_prompt(question: Question, raw_response: str) -> ChatPrompt:
    options = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", question.options))
    user = (f"Options:\n{options}\n\n"
            f"Response to analyze:\n<response>\n{raw_response}\n</response>")
    return ChatPrompt(system=ANALYSIS_SYSTEM, user=user, mode="post_analysis",
                      version=ANALYSIS_PROMPT_VERSION)


@dataclass
class AnalysisSummary:
    original_invalid: int = 0
    already_done: int = 0
    analyzed_now: int = 0
    resolved: int = 0
    unresolved: int = 0
    failed: int = 0
    aborted: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class InvalidAnswerAnalyzer:

    def __init__(self, provider: Optional[BaseLLMProvider], model_id: str):
        self.provider = provider
        self.model_id = model_id

    def analyze(self, raw: dict[str, Any], question: Question) -> dict[str, Any]:
        base = {
            "index": raw["index"],
            "question_type": raw.get("question_type"),
            "original_prediction": raw.get("prediction"),
            "recovered_prediction": None,
            "recovered_letter": None,
            "analysis_status": None,
            "analysis_response": None,
            "evidence": None,
            "failure_reason": None,
            "model_id": self.model_id,
            "input_tokens": 0,
            "output_tokens": 0,
            "analysis_prompt_version": ANALYSIS_PROMPT_VERSION,
            "timestamp": _now(),
        }
        text = (raw.get("raw_response") or "").strip()
        if not text:
            return {**base, "analysis_status": UNRESOLVED, "model_id": None,
                    "failure_reason": f"empty_response (stop_reason={raw.get('stop_reason')})"}

        try:
            response = self.provider.generate(build_analysis_prompt(question, text),
                                              json_schema=ANALYSIS_SCHEMA)
        except FatalLLMError:
            raise
        except Exception as exc:
            return {**base, "analysis_status": FAILED, "failure_reason": f"{type(exc).__name__}: {exc}"[:2000]}

        rec = {**base, "analysis_response": response.text, "model_id": response.model_id,
               "input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        try:
            data = json.loads(response.text)
            choice = data["selected_option"]
            rec["evidence"] = data.get("evidence")
        except (json.JSONDecodeError, KeyError, TypeError):
            return {**rec, "analysis_status": FAILED,
                    "failure_reason": f"unparseable analysis output (stop_reason={response.stop_reason})"}

        if choice in ANSWER_TO_INDEX:
            return {**rec, "analysis_status": RESOLVED, "recovered_prediction": ANSWER_TO_INDEX[choice],
                    "recovered_letter": choice}
        if choice == "UNRESOLVED":
            return {**rec, "analysis_status": UNRESOLVED, "failure_reason": "model_reported_unresolved"}
        return {**rec, "analysis_status": FAILED, "failure_reason": f"unexpected option {choice!r}"}


def run_post_analysis(storage: ExperimentStorage, dataset: ORQADataset, analyzer: Optional[InvalidAnswerAnalyzer],
                      *, reanalyze: bool = False, max_consecutive_errors: int = 5,
                      progress: Optional[Callable[[int, int], None]] = None) -> AnalysisSummary:
    """analyzer=None 代表 post-analysis 停用：只產生合併檔，所有 invalid 標記為 not_analyzed。"""
    raw = storage.raw_predictions()
    invalid = [raw[i] for i in sorted(raw) if raw[i].get("prediction") is None]
    summary = AnalysisSummary(original_invalid=len(invalid))

    if analyzer is not None:
        done = storage.latest(HAIKU_ANALYSIS)
        todo = [r for r in invalid
                if reanalyze or done.get(r["index"], {}).get("analysis_status") not in FINAL_STATUSES]
        summary.already_done = len(invalid) - len(todo)
        consecutive = 0
        for n, r in enumerate(todo, start=1):
            rec = analyzer.analyze(r, dataset.get(r["index"]))
            storage.append(HAIKU_ANALYSIS, rec)
            summary.analyzed_now += 1
            consecutive = consecutive + 1 if rec["analysis_status"] == FAILED else 0
            if progress:
                progress(n, len(todo))
            if consecutive >= max_consecutive_errors:
                summary.aborted = True
                break

    write_post_analysis_files(storage)
    latest = storage.latest(HAIKU_ANALYSIS)
    for r in invalid:
        status = latest.get(r["index"], {}).get("analysis_status")
        if status == RESOLVED:
            summary.resolved += 1
        elif status == UNRESOLVED:
            summary.unresolved += 1
        elif status == FAILED:
            summary.failed += 1
    return summary


def merge_predictions(raw: dict[int, dict], analyses: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """依 index 合併原始預測與 Haiku 分析 → (post-analysis predictions, checked invalid)。"""
    post, checked = [], []
    for idx in sorted(raw):
        r = raw[idx]
        a = analyses.get(idx)
        status = a.get("analysis_status") if a else None
        rec = {
            "index": idx,
            "question_type": r.get("question_type"),
            "raw_prediction": r.get("prediction"),
            "analysis_status": None,
        }
        if r.get("prediction") is not None:
            rec.update(prediction=r["prediction"], prediction_source="raw")
        elif status == RESOLVED:
            rec.update(prediction=a["recovered_prediction"], prediction_source="haiku",
                       analysis_status=RESOLVED)
        else:
            rec.update(prediction=None, prediction_source="invalid",
                       analysis_status=status or "not_analyzed")
            checked.append({
                "index": idx,
                "question_type": r.get("question_type"),
                "original_prediction": r.get("prediction"),
                "raw_response": r.get("raw_response"),
                "analysis_status": status or "not_analyzed",
                "failure_reason": (a or {}).get("failure_reason") or (
                    "post_analysis_disabled_or_not_run" if a is None else None),
                "analysis_response": (a or {}).get("analysis_response"),
            })
        rec["prediction_letter"] = "ABCD"[rec["prediction"]] if rec["prediction"] is not None else None
        rec["is_valid"] = rec["prediction"] is not None
        post.append(rec)
    return post, checked


def write_post_analysis_files(storage: ExperimentStorage) -> tuple[list[dict], list[dict]]:
    post, checked = merge_predictions(storage.raw_predictions(), storage.latest(HAIKU_ANALYSIS))
    storage.write_derived_jsonl(POST_PREDICTIONS, post)
    storage.write_derived_jsonl(CHECKED_INVALID, checked)
    return post, checked
