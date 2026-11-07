"""
orqa/engine.py

單一 Inference Engine：四種模式（basic / skill / rag / skill_rag）× 任何 LLM Provider 共用。

流程（每題）：
    [rag] RAGPipeline.retrieve_for_question → RetrievalResult（寫入 retrievals.jsonl）
    build_prompt(system=指示+skill, user=[檢索內容]+題目)
    provider.generate / generate_batch
    parse_answer → predictions_raw.jsonl（不可變）；Invalid 另寫 invalid_answers.jsonl

Engine 只依賴 BaseLLMProvider 介面，不會 import anthropic / transformers。
Resume：以 index 判斷已完成題目；失敗的題目不寫入 raw，下次執行會重跑。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional, Sequence

from orqa.dataset import Question
from orqa.llm.base import BaseLLMProvider, FatalLLMError, LLMResponse
from orqa.parser import parse_answer
from orqa.prompts import ChatPrompt, build_prompt
from orqa.storage import INFERENCE_ERRORS, INVALID_ANSWERS, RETRIEVALS, ExperimentStorage


class InferenceAborted(RuntimeError):
    pass


@dataclass
class InferenceSummary:
    total: int = 0
    skipped_completed: int = 0
    written: int = 0
    invalid: int = 0
    errors: int = 0
    error_indices: list[int] = field(default_factory=list)
    aborted: bool = False
    abort_reason: Optional[str] = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["error_indices"] = sorted(set(self.error_indices))
        return d


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _batches(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


class InferenceEngine:

    def __init__(self, provider: BaseLLMProvider, mode: str, *, skill=None, rag=None,
                 experiment_id: Optional[str] = None):
        if mode in ("skill", "skill_rag") and skill is None:
            raise ValueError(f"mode={mode} requires a skill")
        if mode in ("rag", "skill_rag") and rag is None:
            raise ValueError(f"mode={mode} requires a RAG pipeline")
        self.provider = provider
        self.mode = mode
        self.skill = skill if mode in ("skill", "skill_rag") else None
        self.rag = rag if mode in ("rag", "skill_rag") else None
        self.experiment_id = experiment_id

    # ------------------------------------------------------------ building

    def prepare(self, question: Question):
        retrieval = self.rag.retrieve_for_question(question) if self.rag is not None else None
        prompt = build_prompt(question, self.mode, skill=self.skill,
                              retrieved_chunks=retrieval.texts if retrieval is not None else None)
        return prompt, retrieval

    def make_record(self, question: Question, prompt: ChatPrompt, response: LLMResponse,
                    retrieval=None) -> dict:
        parsed = parse_answer(response.text)
        record = {
            "index": question.index,
            "question_type": question.question_type,
            "prediction": parsed.prediction,           # None = Invalid
            "prediction_letter": parsed.letter,
            "target_answer": question.target_answer,
            "target_answer_letter": "ABCD"[question.target_answer],
            "is_valid": parsed.is_valid,
            "parse_method": parsed.method,
            "raw_response": response.text,
            "model_id": response.model_id,
            "provider": response.provider,
            "inference_mode": self.mode,
            "prompt_version": prompt.version,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cache_creation_input_tokens": response.cache_creation_input_tokens,
            "cache_read_input_tokens": response.cache_read_input_tokens,
            "stop_reason": response.stop_reason,
            "latency_seconds": round(response.latency_seconds, 3),
            "experiment_id": self.experiment_id,
            "timestamp": _now(),
        }
        if response.thinking:
            record["thinking"] = response.thinking
        if response.extra:
            record["provider_extra"] = response.extra
        if retrieval is not None:
            record["retrieved_chunk_ids"] = [c.chunk_id for c in retrieval.chunks]
        return record

    # ----------------------------------------------------------------- run

    def run(self, questions: Sequence[Question], storage: ExperimentStorage, *,
            max_consecutive_errors: int = 5,
            progress: Optional[Callable[[int], None]] = None) -> InferenceSummary:
        completed = storage.completed_indices()
        todo = [q for q in questions if q.index not in completed]
        summary = InferenceSummary(total=len(questions), skipped_completed=len(questions) - len(todo))
        consecutive = 0

        for batch in _batches(todo, max(1, self.provider.preferred_batch_size)):
            try:
                prepared = [(q, *self.prepare(q)) for q in batch]
                prompts = [p for _, p, _ in prepared]
                responses = (self.provider.generate_batch(prompts) if len(prompts) > 1
                             else [self.provider.generate(prompts[0])])
                if len(responses) != len(prepared):
                    raise RuntimeError("provider returned a different number of responses")
            except FatalLLMError:
                raise
            except Exception as exc:  # 單一 batch 失敗：記錄後繼續，達上限才中止
                consecutive += 1
                summary.errors += len(batch)
                summary.error_indices.extend(q.index for q in batch)
                storage.append(INFERENCE_ERRORS, {
                    "indices": [q.index for q in batch],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "traceback": traceback.format_exc(limit=5)[-4000:],
                    "timestamp": _now(),
                })
                if progress:
                    progress(len(batch))
                if consecutive >= max_consecutive_errors:
                    summary.aborted = True
                    summary.abort_reason = f"{consecutive} consecutive failed batches; last error: {exc}"
                    raise InferenceAborted(summary.abort_reason) from exc
                continue

            consecutive = 0
            for (q, prompt, retrieval), response in zip(prepared, responses):
                record = self.make_record(q, prompt, response, retrieval)
                if retrieval is not None:
                    storage.append(RETRIEVALS, {"index": q.index, "question_type": q.question_type,
                                                **retrieval.to_record(include_content=True)})
                storage.append_raw_prediction(record, completed=completed)
                summary.written += 1
                if not record["is_valid"]:
                    summary.invalid += 1
                    storage.append(INVALID_ANSWERS, {
                        "index": q.index,
                        "question_type": q.question_type,
                        "prediction": None,
                        "target_answer": q.target_answer,
                        "target_answer_letter": "ABCD"[q.target_answer],
                        "raw_response": response.text,
                        "stop_reason": response.stop_reason,
                        "model_id": response.model_id,
                        "timestamp": record["timestamp"],
                    })
            if progress:
                progress(len(batch))
        return summary
