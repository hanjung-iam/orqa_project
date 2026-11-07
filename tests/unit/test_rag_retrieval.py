"""
RAG retrieval tests（不呼叫 Voyage API）：
- FakeEmbedder（bag-of-words hashing）+ 真實 ChromaDB（暫存目錄）
- BM25 / RRF 與 legacy Colab 實作比對
- index 版本控管、續建、threshold 行為
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from pathlib import Path

import pytest

from orqa.config import (ChunkingSection, DocumentSpec, RAGSection, RetrievalSection,
                         VectorStoreSection)
from orqa.dataset import Question
from orqa.rag.embedding import BaseEmbedder, EmbeddingError
from orqa.rag.pipeline import IndexNotReadyError, RAGPipeline, build_retrieval_query
from orqa.rag.retrieval import Candidate, reciprocal_rank_fusion, tokenize_for_bm25

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "legacy" / "rag_colab_reference.py"
DIM = 64


class FakeEmbedder(BaseEmbedder):
    def __init__(self):
        super().__init__("fake-embed")
        self.inputs: list[tuple[str, int]] = []

    def _embed(self, texts, input_type):
        self.inputs.append((input_type, len(texts)))
        out = []
        for t in texts:
            v = [0.0] * DIM
            for tok in tokenize_for_bm25(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        self.calls += 1
        return out


KB_CASES = """## CASE c1
**Model class:** LP
### Scenario
A bakery decides how many loaves of bread and cakes to bake to maximize profit under oven capacity.

## CASE c2
**Model class:** MILP
### Scenario
A logistics firm opens warehouses with binary open variables and assigns customers to warehouses.

## CASE c3
**Model class:** ILP
### Scenario
Nurse scheduling assigns nurses to shifts subject to coverage constraints each day.
"""

KB_BOOK = """# Linear Programming
Linear programming optimizes a linear objective subject to linear constraints.

## Integer Programming
Binary variables model yes/no decisions such as opening a facility.

## Network Flow
Flow conservation constraints balance inflow and outflow at each node.
"""


@pytest.fixture
def kb(tmp_path):
    (tmp_path / "StructuredOR.md").write_text(KB_CASES, encoding="utf-8")
    (tmp_path / "book.md").write_text(KB_BOOK, encoding="utf-8")
    return tmp_path


def make_settings(kb: Path, tmp_path: Path, **retrieval) -> RAGSection:
    return RAGSection(
        documents=(DocumentSpec(str(kb / "StructuredOR.md"), "case_aware"),
                   DocumentSpec(str(kb / "book.md"), "heading_based")),
        vector_store=VectorStoreSection(persist_directory=str(tmp_path / "vdb"), collection_prefix="test_kb"),
        chunking=ChunkingSection(chunk_size=2048, stride=256),
        retrieval=dataclasses.replace(RetrievalSection(), **retrieval),
    )


def test_build_index_and_hybrid_retrieve(kb, tmp_path):
    emb = FakeEmbedder()
    p = RAGPipeline(make_settings(kb, tmp_path, similarity_threshold=0.0, final_top_k=3), embedder=emb)
    assert p.index_status()["status"] == "missing"
    with pytest.raises(IndexNotReadyError):
        p.retrieve("warehouse")

    built = p.build_index()
    assert built["status"] == "complete" and built["embedded_now"] == len(p.chunks) == 6
    assert all(t == "document" for t, _ in emb.inputs)

    r = p.retrieve("open warehouses with binary variables")
    assert r.mode == "hybrid" and len(r.chunks) == 3
    assert emb.inputs[-1] == ("query", 1)                       # 查詢使用 input_type=query
    top = r.chunks[0]
    assert top.chunk_id == "StructuredOR.md::00001"             # warehouse case
    assert top.metadata["case_id"] == "c2" and top.metadata["embedding_model"] == "voyage-4-large"
    assert top.dense_rank is not None and top.bm25_rank is not None
    assert top.rrf_score == pytest.approx(1 / (60 + top.dense_rank) + 1 / (60 + top.bm25_rank))
    assert [c.rank for c in r.chunks] == [1, 2, 3]
    assert all(r.chunks[i].score >= r.chunks[i + 1].score for i in range(2))

    # 第二次 build：index 已完成，不再 embedding
    calls = emb.calls
    again = p.build_index()
    assert again["embedded_now"] == 0 and emb.calls == calls


def test_dense_threshold_applies_only_to_dense(kb, tmp_path):
    emb = FakeEmbedder()
    p = RAGPipeline(make_settings(kb, tmp_path, similarity_threshold=0.99), embedder=emb)
    p.build_index()
    r = p.retrieve("nurse shifts coverage")
    assert r.dense_candidates == 0                    # 全部低於 0.99 → dense 為空
    assert r.bm25_candidates > 0 and len(r.chunks) > 0  # BM25 不受 threshold 影響
    assert all(c.dense_rank is None and c.bm25_rank is not None for c in r.chunks)


def test_similarity_is_one_minus_cosine_distance(kb, tmp_path):
    emb = FakeEmbedder()
    p = RAGPipeline(make_settings(kb, tmp_path, mode="dense", similarity_threshold=0.0), embedder=emb)
    p.build_index()
    q = "binary variables model yes/no decisions such as opening a facility"
    r = p.retrieve(q)
    qv = emb.embed_query(q)
    for c in r.chunks:
        dv = emb.embed_documents([c.content])[0]
        cos = sum(a * b for a, b in zip(qv, dv))
        assert c.dense_similarity == pytest.approx(cos, abs=1e-4)
        assert c.dense_distance == pytest.approx(1 - cos, abs=1e-4)
    sims = [c.dense_similarity for c in r.chunks]
    assert sims == sorted(sims, reverse=True)


def test_bm25_mode_needs_no_vector_index(kb, tmp_path):
    class Boom(BaseEmbedder):
        def _embed(self, texts, input_type):
            raise AssertionError("BM25 mode must not call the embedder")
    p = RAGPipeline(make_settings(kb, tmp_path, mode="bm25", final_top_k=2), embedder=Boom("x"))
    r = p.retrieve("bread cakes oven")
    assert r.chunks[0].metadata["case_id"] == "c1" and r.chunks[0].bm25_score > 0
    assert r.chunks[0].rrf_score is None


def test_metadata_filter(kb, tmp_path):
    p = RAGPipeline(make_settings(kb, tmp_path, similarity_threshold=0.0), embedder=FakeEmbedder())
    p.build_index()
    r = p.retrieve("binary variables warehouses", where={"document_name": "book.md"})
    assert r.chunks and all(c.metadata["document_name"] == "book.md" for c in r.chunks)


def test_index_key_changes_with_settings_and_documents(kb, tmp_path):
    a = RAGPipeline(make_settings(kb, tmp_path), embedder=FakeEmbedder())
    b = RAGPipeline(dataclasses.replace(make_settings(kb, tmp_path),
                                        chunking=ChunkingSection(chunk_size=1024, stride=128)),
                    embedder=FakeEmbedder())
    assert a.collection_name != b.collection_name
    (kb / "book.md").write_text(KB_BOOK + "\n## New\nMore text.\n", encoding="utf-8")
    c = RAGPipeline(make_settings(kb, tmp_path), embedder=FakeEmbedder())
    assert c.collection_name != a.collection_name


def test_resume_partial_build(kb, tmp_path):
    class FlakyEmbedder(FakeEmbedder):
        fail = True
        def _embed(self, texts, input_type):
            if self.fail and input_type == "document" and self.calls >= 1:
                raise EmbeddingError("simulated outage")
            return super()._embed(texts, input_type)

    settings = make_settings(kb, tmp_path)
    settings = dataclasses.replace(settings, embedding=dataclasses.replace(settings.embedding, batch_size=2))
    emb = FlakyEmbedder()
    p = RAGPipeline(settings, embedder=emb)
    with pytest.raises(EmbeddingError):
        p.build_index()
    assert p.index_status()["status"] == "building" and p.store.count() == 2

    emb.fail = False
    p2 = RAGPipeline(settings, embedder=emb)
    result = p2.build_index()
    assert result["status"] == "complete" and result["embedded_now"] == 4  # 只補缺的 4 個


def test_retrieval_query_excludes_options():
    q = Question(0, "Q1", "ctx text", "what?", ("OPT_A", "OPT_B", "OPT_C", "OPT_D"), 0)
    query = build_retrieval_query(q)
    assert query == "Context:\nctx text\n\nQuestion:\nwhat?"
    assert "OPT_" not in query


# ------------------------------------------------------ legacy equivalence

def _exec_legacy(start_marker: str, end_marker: str, ns: dict) -> dict:
    src = LEGACY.read_text(encoding="utf-8")
    code = src[src.index(start_marker): src.index(end_marker)]
    exec(compile(code, str(LEGACY), "exec"), ns)
    return ns


def test_rrf_matches_legacy():
    ns = _exec_legacy("def reciprocal_rank_fusion", "def retrieve_hybrid_chunks", {"RRF_K": 60, "FINAL_TOP_K": 3})
    dense = [Candidate(f"c{i}", r, 0.9) for r, i in enumerate([3, 1, 7, 5], start=1)]
    bm25 = [Candidate(f"c{i}", r, 5.0) for r, i in enumerate([1, 9, 3, 2, 7], start=1)]
    legacy_dense = [{"rank": c.rank, "metadata": {"chunk_id": c.chunk_id}, "content": ""} for c in dense]
    legacy_bm25 = [{"rank": c.rank, "chunk_id": c.chunk_id, "content": "", "metadata": {}} for c in bm25]
    old = ns["reciprocal_rank_fusion"](legacy_dense, legacy_bm25, rrf_k=60, final_top_k=5)
    new = reciprocal_rank_fusion(dense, bm25, rrf_k=60, final_top_k=5)
    assert [(o["chunk_id"], o["rrf_score"]) for o in old] == new
    assert new[0][0] in ("c1", "c3")  # 同時出現在兩邊的 chunk 分數最高


def test_bm25_matches_legacy(kb, tmp_path):
    from orqa.rag.retrieval import BM25Retriever
    p = RAGPipeline(make_settings(kb, tmp_path, mode="bm25"), embedder=FakeEmbedder())
    legacy_chunks = [{"chunk_id": c.chunk_id, "content": c.content} for c in p.chunks]
    ns = _exec_legacy("from rank_bm25 import BM25Okapi", "bm25_test_results =",
                      {"re": re, "chunks": legacy_chunks, "BM25_TOP_K": 15})
    query = "binary variables decide to open warehouses"
    old = ns["retrieve_bm25_chunks"](query, top_k=5)
    new = BM25Retriever(p.chunks).retrieve(query, top_k=5)
    assert [(o["chunk_id"], o["rank"], o["score"]) for o in old] == [(c.chunk_id, c.rank, c.score) for c in new]
