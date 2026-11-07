"""
orqa/rag/embedding.py

Voyage Embedding。

- API Key：環境變數 VOYAGE_API_KEY
- 文件使用 input_type="document"，查詢使用 input_type="query"
- truncation=False：輸入過長時直接報錯，不會靜默截斷
- 自行處理 rate limit / 暫時性錯誤的 exponential backoff 重試；
  單一 batch 超過 API 限制（InvalidRequestError）時自動對半拆分
- 檢查 embedding 數量與輸入數量一致、維度一致
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from orqa.config import EmbeddingSection, get_secret


class EmbeddingError(RuntimeError):
    pass


class BaseEmbedder(ABC):

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.dimension: Optional[int] = None
        self.total_tokens = 0
        self.calls = 0

    @abstractmethod
    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        ...

    def _checked(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors = self._embed(texts, input_type)
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Embedding count mismatch: {len(vectors)} vectors for {len(texts)} texts")
        for v in vectors:
            if self.dimension is None:
                self.dimension = len(v)
            elif len(v) != self.dimension:
                raise EmbeddingError(f"Embedding dimension mismatch: {len(v)} != {self.dimension}")
        return vectors

    def embed_documents(self, texts: Sequence[str], batch_size: int = 128,
                        on_batch: Optional[Callable[[int, int], None]] = None) -> list[list[float]]:
        texts = list(texts)
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            out.extend(self._checked(texts[start:start + batch_size], "document"))
            if on_batch:
                on_batch(min(start + batch_size, len(texts)), len(texts))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._checked([text], "query")[0]

    def info(self) -> dict[str, Any]:
        return {"model": self.model_id, "dimension": self.dimension,
                "total_tokens": self.total_tokens, "calls": self.calls}


class VoyageEmbedder(BaseEmbedder):

    def __init__(self, settings: EmbeddingSection, *, client: Any = None):
        super().__init__(settings.model)
        self.settings = settings
        import voyageai
        import voyageai.error as verr
        self._errors = verr
        # 重試由本類別處理，client 端關閉以免重複重試
        self.client = client or voyageai.Client(api_key=get_secret("VOYAGE_API_KEY"), max_retries=0)

    def _call(self, texts: list[str], input_type: str):
        kwargs: dict[str, Any] = {
            "texts": texts,
            "model": self.settings.model,
            "input_type": input_type,
            "truncation": False,
        }
        if self.settings.output_dimension is not None:
            kwargs["output_dimension"] = self.settings.output_dimension

        e = self._errors
        retryable = (e.RateLimitError, e.ServiceUnavailableError, e.ServerError,
                     e.APIConnectionError, e.APIError)
        fatal = (e.AuthenticationError, e.MalformedRequestError)
        for attempt in range(self.settings.max_retries + 1):
            try:
                return self.client.embed(**kwargs)
            except fatal as exc:
                raise EmbeddingError(f"Voyage API rejected the request: {exc}") from exc
            except e.InvalidRequestError:
                raise
            except retryable as exc:
                if attempt >= self.settings.max_retries:
                    raise EmbeddingError(f"Voyage API failed after {attempt + 1} attempts: {exc}") from exc
                time.sleep(min(60.0, 2 ** attempt + random.uniform(0, 1)))
        raise AssertionError("unreachable")

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            result = self._call(texts, input_type)
        except self._errors.InvalidRequestError as exc:
            if len(texts) == 1:
                raise EmbeddingError(f"Voyage rejected a single input (too long?): {exc}") from exc
            mid = len(texts) // 2  # 例如 batch token 數超過上限 → 對半拆分
            return self._embed(texts[:mid], input_type) + self._embed(texts[mid:], input_type)
        self.calls += 1
        self.total_tokens += int(getattr(result, "total_tokens", 0) or 0)
        return [list(map(float, v)) for v in result.embeddings]

    def info(self) -> dict[str, Any]:
        return {**super().info(), "provider": "voyage", "output_dimension": self.settings.output_dimension}
