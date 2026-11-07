"""測試用的 fake provider / fake RAG（不呼叫任何外部服務）。"""

from __future__ import annotations

from typing import Callable, Optional

from orqa.llm.base import BaseLLMProvider, LLMResponse
from orqa.prompts import ChatPrompt
from orqa.rag.retrieval import RetrievalResult, RetrievedChunk


class FakeProvider(BaseLLMProvider):
    provider_name = "fake"

    def __init__(self, answer: Callable[[ChatPrompt], str] | str = "A", *, batch_size: int = 1,
                 fail_on: Optional[Callable[[ChatPrompt], bool]] = None, model_id: str = "fake-model"):
        super().__init__()
        self.answer = answer
        self.batch_size = batch_size
        self.fail_on = fail_on
        self.model_id = model_id
        self.prompts: list[ChatPrompt] = []
        self.batch_calls = 0

    def _generate(self, prompt, *, json_schema=None):
        self.prompts.append(prompt)
        if self.fail_on and self.fail_on(prompt):
            raise RuntimeError("simulated provider failure")
        text = self.answer(prompt) if callable(self.answer) else self.answer
        return LLMResponse(text=text, model_id=self.model_id, provider=self.provider_name,
                           input_tokens=len(prompt.user), output_tokens=len(text), stop_reason="end_turn")

    def generate_batch(self, prompts):
        self.batch_calls += 1
        return [self.generate(p) for p in prompts]

    @property
    def preferred_batch_size(self) -> int:
        return self.batch_size

    def get_model_info(self):
        return {"provider": "fake", "model_id": self.model_id, "quantization": None,
                "generation_parameters": {"max_tokens": 16}}


class FakeRAG:
    """回傳固定的 2 個 chunk，內容帶有題目 index 方便驗證。"""

    def __init__(self):
        self.queries: list[int] = []

    def retrieve_for_question(self, question) -> RetrievalResult:
        self.queries.append(question.index)
        chunks = tuple(
            RetrievedChunk(chunk_id=f"doc::{question.index:03d}{k}", content=f"KNOWLEDGE-{question.index}-{k}",
                           metadata={"document_name": "doc.md", "chunk_strategy": "heading_based"},
                           rank=k + 1, score=0.03 - k * 0.001, dense_rank=k + 1, dense_similarity=0.8,
                           dense_distance=0.2, bm25_rank=k + 2, bm25_score=12.0, rrf_score=0.03 - k * 0.001)
            for k in range(2)
        )
        return RetrievalResult(query=f"q{question.index}", mode="hybrid", chunks=chunks,
                               params={"mode": "hybrid"}, dense_candidates=2, bm25_candidates=2)

    def describe(self):
        return {"index_key": "fake-index", "embedding_model": "fake-embed", "retrieval_mode": "hybrid"}
