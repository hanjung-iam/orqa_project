"""
orqa/rag/retrieval.py

Dense / BM25 / Hybrid(RRF) Retrieval —— 邏輯與 legacy/rag_colab_reference.py 相同，
差異只在 Dense 於 Hybrid 中也套用 similarity_threshold（舊版 hybrid 傳入 0）。

- Dense ：ChromaDB cosine distance → similarity = 1 - distance；similarity < threshold 的結果排除。
          rank 為 Chroma 原始排序（排除只會發生在尾端，所以 rank 不變）。
- BM25  ：rank-bm25 BM25Okapi；tokenize = 小寫 + \\b\\w+\\b；保留原始分數（包含 0 分，與舊版相同）。
- RRF   ：score(d) = Σ 1 / (rrf_k + rank_i(d))，只使用 rank，不混用 dense 與 BM25 的原始分數。
          不套用 similarity threshold。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from orqa.rag.chunking import Chunk


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    rank: int
    score: float                    # dense: similarity；bm25: BM25 score
    distance: Optional[float] = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    rank: int                       # 最終排名（1-based）
    score: float                    # 最終排序依據：hybrid=RRF，dense=similarity，bm25=BM25 score
    dense_rank: Optional[int] = None
    dense_similarity: Optional[float] = None
    dense_distance: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None

    def to_record(self, include_content: bool = True) -> dict[str, Any]:
        rec = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "dense_similarity": self.dense_similarity,
            "dense_distance": self.dense_distance,
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "document_name": self.metadata.get("document_name"),
            "chunk_strategy": self.metadata.get("chunk_strategy"),
            "case_id": self.metadata.get("case_id") or None,
            "heading_path": self.metadata.get("heading_path") or None,
        }
        if include_content:
            rec["content"] = self.content
        return rec


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    mode: str
    chunks: tuple[RetrievedChunk, ...]
    params: dict[str, Any] = field(default_factory=dict)
    dense_candidates: int = 0
    bm25_candidates: int = 0

    @property
    def texts(self) -> list[str]:
        return [c.content for c in self.chunks]

    def to_record(self, include_content: bool = True) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "query": self.query,
            "params": self.params,
            "dense_candidates": self.dense_candidates,
            "bm25_candidates": self.bm25_candidates,
            "chunks": [c.to_record(include_content) for c in self.chunks],
        }


class BM25Retriever:

    def __init__(self, chunks: Sequence[Chunk]):
        from rank_bm25 import BM25Okapi
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("BM25 corpus is empty")
        self.bm25 = BM25Okapi([tokenize_for_bm25(c.content) for c in self.chunks])

    def retrieve(self, query: str, top_k: int) -> list[Candidate]:
        scores = self.bm25.get_scores(tokenize_for_bm25(query))
        # sorted 為 stable sort：同分時保留 corpus 順序（與舊版相同）
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Candidate(self.chunks[i].chunk_id, rank, float(scores[i]))
                for rank, i in enumerate(order, start=1)]


class DenseRetriever:

    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int, similarity_threshold: float,
                 where: Optional[dict] = None) -> list[Candidate]:
        hits = self.store.query(self.embedder.embed_query(query), n_results=top_k, where=where)
        out = []
        for rank, hit in enumerate(hits, start=1):
            if hit.similarity < similarity_threshold:
                continue
            out.append(Candidate(hit.chunk_id, rank, hit.similarity, hit.distance))
        return out


def reciprocal_rank_fusion(dense: Sequence[Candidate], bm25: Sequence[Candidate],
                           rrf_k: int, final_top_k: int) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for c in dense:
        fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (rrf_k + c.rank)
    for c in bm25:
        fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (rrf_k + c.rank)
    ranked = sorted(fused, key=fused.get, reverse=True)[:final_top_k]
    return [(cid, fused[cid]) for cid in ranked]


def assemble(mode: str, query: str, lookup: dict[str, Chunk], metadata_of,
             dense: Sequence[Candidate], bm25: Sequence[Candidate],
             rrf_k: int, final_top_k: int, params: dict) -> RetrievalResult:
    d_by_id = {c.chunk_id: c for c in dense}
    b_by_id = {c.chunk_id: c for c in bm25}

    if mode == "hybrid":
        order = reciprocal_rank_fusion(dense, bm25, rrf_k, final_top_k)
    elif mode == "dense":
        order = [(c.chunk_id, c.score) for c in dense[:final_top_k]]
    elif mode == "bm25":
        order = [(c.chunk_id, c.score) for c in bm25[:final_top_k]]
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    chunks = []
    for final_rank, (cid, score) in enumerate(order, start=1):
        d, b = d_by_id.get(cid), b_by_id.get(cid)
        chunks.append(RetrievedChunk(
            chunk_id=cid,
            content=lookup[cid].content,
            metadata=metadata_of(lookup[cid]),
            rank=final_rank,
            score=score,
            dense_rank=d.rank if d else None,
            dense_similarity=d.score if d else None,
            dense_distance=d.distance if d else None,
            bm25_rank=b.rank if b else None,
            bm25_score=b.score if b else None,
            rrf_score=score if mode == "hybrid" else None,
        ))
    return RetrievalResult(query=query, mode=mode, chunks=tuple(chunks), params=params,
                           dense_candidates=len(dense), bm25_candidates=len(bm25))
