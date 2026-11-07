"""VoyageEmbedder 介面測試（fake client，不呼叫 API）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import voyageai.error as verr

from orqa.config import EmbeddingSection
from orqa.rag.embedding import EmbeddingError, VoyageEmbedder


class FakeVoyage:
    def __init__(self, max_batch=None, dim=4, fail_times=0, wrong_count=False):
        self.calls = []
        self.max_batch = max_batch
        self.dim = dim
        self.fail_times = fail_times
        self.wrong_count = wrong_count

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times:
            self.fail_times -= 1
            raise verr.RateLimitError("slow down")
        texts = kwargs["texts"]
        if self.max_batch and len(texts) > self.max_batch:
            raise verr.InvalidRequestError("too many tokens in batch")
        n = len(texts) - 1 if self.wrong_count else len(texts)
        return SimpleNamespace(embeddings=[[0.5] * self.dim for _ in range(n)], total_tokens=10 * n)


def make(client, **kw):
    return VoyageEmbedder(EmbeddingSection(**kw), client=client)


def test_document_and_query_input_types():
    fake = FakeVoyage()
    e = make(fake, output_dimension=1024)
    assert len(e.embed_documents(["a", "b", "c"], batch_size=2)) == 3
    e.embed_query("q")
    assert [c["input_type"] for c in fake.calls] == ["document", "document", "query"]
    assert all(c["model"] == "voyage-4-large" and c["truncation"] is False for c in fake.calls)
    assert all(c["output_dimension"] == 1024 for c in fake.calls)
    assert e.dimension == 4 and e.total_tokens == 40


def test_oversized_batch_is_split():
    fake = FakeVoyage(max_batch=2)
    vectors = make(fake).embed_documents([str(i) for i in range(5)], batch_size=5)
    assert len(vectors) == 5
    succeeded = [c for c in fake.calls if len(c["texts"]) <= 2]
    assert sum(len(c["texts"]) for c in succeeded) == 5   # 每段文字都只成功 embedding 一次


def test_rate_limit_retry(monkeypatch):
    monkeypatch.setattr("orqa.rag.embedding.time.sleep", lambda s: None)
    fake = FakeVoyage(fail_times=2)
    assert len(make(fake, max_retries=3).embed_documents(["x"])) == 1
    assert len(fake.calls) == 3


def test_retry_exhausted(monkeypatch):
    monkeypatch.setattr("orqa.rag.embedding.time.sleep", lambda s: None)
    with pytest.raises(EmbeddingError, match="after 2 attempts"):
        make(FakeVoyage(fail_times=5), max_retries=1).embed_documents(["x"])


def test_count_mismatch_detected():
    with pytest.raises(EmbeddingError, match="count mismatch"):
        make(FakeVoyage(wrong_count=True)).embed_documents(["a", "b"])
