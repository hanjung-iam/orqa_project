"""
orqa/rag/pipeline.py

RAG Pipeline：文件 → chunking → Voyage embedding → ChromaDB；查詢時執行 Dense / BM25 / Hybrid。

Index 版本控管
--------------
index_key = SHA256(chunker 版本、chunk_size、stride、embedding model、output_dimension、
                   每份文件的名稱 + 內容 SHA256 + chunking 策略)
collection 名稱 = <collection_prefix>_<index_key 前 16 碼>

文件內容、chunking 設定或 embedding model 任何一項改變，都會對應到新的 collection，
不會把不同版本的 embedding 混在同一個 collection。舊 collection 保留，不會自動刪除。

建立 index 時若中斷，重新執行會沿用已寫入的 chunk（chunk_id 固定），只 embedding 缺少的部分。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Callable, Optional

from orqa.config import RAGSection
from orqa.dataset import Question
from orqa.rag.chunking import CHUNKER_VERSION, Chunk, chunk_document
from orqa.rag.documents import KnowledgeDocument, load_documents
from orqa.rag.retrieval import BM25Retriever, DenseRetriever, RetrievalResult, assemble


class IndexNotReadyError(RuntimeError):
    pass


def build_retrieval_query(question: Question) -> str:
    """檢索 query 只包含 Context 與 Question，不含選項（與 Colab 版相同）。"""
    return f"Context:\n{question.context}\n\nQuestion:\n{question.question}"


class RAGPipeline:

    def __init__(self, settings: RAGSection, *, embedder=None, store=None):
        self.settings = settings
        self.documents: list[KnowledgeDocument] = load_documents(settings.documents)
        c = settings.chunking
        self.chunks: list[Chunk] = [
            chunk
            for doc in self.documents
            for chunk in chunk_document(doc.name, doc.content, doc.strategy, c.chunk_size, c.stride)
        ]
        self.lookup = {ch.chunk_id: ch for ch in self.chunks}
        if len(self.lookup) != len(self.chunks):
            raise RuntimeError("Duplicate chunk_id produced by chunking")
        self._doc_by_name = {d.name: d for d in self.documents}

        self.index_spec = {
            "chunker_version": CHUNKER_VERSION,
            "chunk_size": c.chunk_size,
            "stride": c.stride,
            "embedding_provider": settings.embedding.provider,
            "embedding_model": settings.embedding.model,
            "output_dimension": settings.embedding.output_dimension,
            "documents": [{"name": d.name, "sha256": d.sha256, "strategy": d.strategy}
                          for d in self.documents],
        }
        self.index_key = hashlib.sha256(
            json.dumps(self.index_spec, sort_keys=True).encode("utf-8")).hexdigest()
        self.collection_name = f"{settings.vector_store.collection_prefix}_{self.index_key[:16]}"

        self._embedder = embedder
        self._store = store
        self._bm25: Optional[BM25Retriever] = None
        self._dense: Optional[DenseRetriever] = None

    # ----------------------------------------------------------- lazy parts

    @property
    def needs_dense(self) -> bool:
        return self.settings.retrieval.mode in ("dense", "hybrid")

    @property
    def embedder(self):
        if self._embedder is None:
            from orqa.rag.embedding import VoyageEmbedder
            self._embedder = VoyageEmbedder(self.settings.embedding)
        return self._embedder

    @property
    def store(self):
        if self._store is None:
            from orqa.rag.vector_store import ChromaVectorStore
            self._store = ChromaVectorStore(self.settings.vector_store.persist_directory)
        return self._store

    def chunk_metadata(self, chunk: Chunk) -> dict[str, Any]:
        doc = self._doc_by_name[chunk.document_name]
        return chunk.metadata(source_version=doc.source_version,
                              embedding_model=self.settings.embedding.model,
                              index_key=self.index_key[:16])

    # ---------------------------------------------------------------- index

    def index_status(self) -> dict[str, Any]:
        store = self.store
        if not store.exists(self.collection_name):
            return {"collection": self.collection_name, "exists": False, "status": "missing"}
        store.open(self.collection_name)
        meta = store.metadata
        count = store.count()
        ready = (meta.get("status") == "complete" and count == len(self.chunks)
                 and meta.get("index_key") == self.index_key)
        return {"collection": self.collection_name, "exists": True,
                "status": "complete" if ready else meta.get("status", "unknown"),
                "count": count, "expected": len(self.chunks), "metadata": meta}

    def build_index(self, *, force: bool = False,
                    progress: Optional[Callable[[int, int], None]] = None) -> dict[str, Any]:
        store = self.store
        if force and store.exists(self.collection_name):
            store.delete_collection(self.collection_name)

        status = self.index_status()
        if status["status"] == "complete":
            return {**status, "embedded_now": 0}

        store.open(self.collection_name, create=True, metadata={
            "index_key": self.index_key,
            "status": "building",
            "embedding_model": self.settings.embedding.model,
            "chunker_version": CHUNKER_VERSION,
            "chunk_size": self.settings.chunking.chunk_size,
            "stride": self.settings.chunking.stride,
            "spec_json": json.dumps(self.index_spec, sort_keys=True),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

        expected = {c.chunk_id for c in self.chunks}
        existing = store.ids()
        stale = existing - expected
        if stale:  # 同一個 index_key 不應該出現，保險起見清除
            store.delete(ids=stale)
        missing = [c for c in self.chunks if c.chunk_id not in existing]

        batch = self.settings.embedding.batch_size
        done = 0
        for start in range(0, len(missing), batch):
            part = missing[start:start + batch]
            vectors = self.embedder.embed_documents([c.content for c in part], batch_size=batch)
            store.upsert(ids=[c.chunk_id for c in part], embeddings=vectors,
                         documents=[c.content for c in part],
                         metadatas=[self.chunk_metadata(c) for c in part])
            done += len(part)
            if progress:
                progress(done, len(missing))

        count = store.count()
        if count != len(self.chunks) or store.ids() != expected:
            raise RuntimeError(f"Index verification failed: {count} vectors, expected {len(self.chunks)}")

        dim = self.embedder.dimension or store.metadata.get("embedding_dimension")
        store.update_metadata(status="complete", chunk_count=count,
                              embedding_dimension=int(dim) if dim else -1,
                              completed_at=datetime.now().isoformat(timespec="seconds"))
        return {**self.index_status(), "embedded_now": len(missing),
                "embedding_tokens": self.embedder.total_tokens}

    # ------------------------------------------------------------ retrieval

    def open(self) -> None:
        """準備 retriever。Dense / Hybrid 需要已完成的 index。"""
        if self._bm25 is None and self.settings.retrieval.mode in ("bm25", "hybrid"):
            self._bm25 = BM25Retriever(self.chunks)
        if self._dense is None and self.needs_dense:
            status = self.index_status()
            if status["status"] != "complete":
                raise IndexNotReadyError(
                    f"Vector index {self.collection_name} is not ready ({status['status']}). "
                    "Build it first: python -m orqa.rag build --config <config.yaml>"
                )
            self._dense = DenseRetriever(self.store, self.embedder)

    def retrieve(self, query: str, *, where: Optional[dict] = None) -> RetrievalResult:
        self.open()
        r = self.settings.retrieval
        dense = (self._dense.retrieve(query, r.dense_top_k, r.similarity_threshold, where=where)
                 if self.needs_dense else [])
        bm25 = self._bm25.retrieve(query, r.bm25_top_k) if self._bm25 is not None else []
        if where is not None and bm25:
            bm25 = [c for c in bm25 if _matches(self.chunk_metadata(self.lookup[c.chunk_id]), where)]
        params = {"mode": r.mode, "similarity_threshold": r.similarity_threshold,
                  "dense_top_k": r.dense_top_k, "bm25_top_k": r.bm25_top_k,
                  "final_top_k": r.final_top_k, "rrf_k": r.rrf_k}
        return assemble(r.mode, query, self.lookup, self.chunk_metadata, dense, bm25,
                        r.rrf_k, r.final_top_k, params)

    def retrieve_for_question(self, question: Question) -> RetrievalResult:
        return self.retrieve(build_retrieval_query(question))

    def describe(self) -> dict[str, Any]:
        r = self.settings.retrieval
        e = self.settings.embedding
        return {
            "embedding_model": e.model,
            "embedding_provider": e.provider,
            "embedding_dimension": (self._embedder.dimension if self._embedder else None),
            "vector_store": "chromadb",
            "persist_directory": self.settings.vector_store.persist_directory,
            "collection": self.collection_name,
            "index_key": self.index_key,
            "chunker_version": CHUNKER_VERSION,
            "chunk_size": self.settings.chunking.chunk_size,
            "stride": self.settings.chunking.stride,
            "chunk_count": len(self.chunks),
            "documents": self.index_spec["documents"],
            "retrieval_mode": r.mode,
            "similarity_threshold": r.similarity_threshold,
            "dense_top_k": r.dense_top_k,
            "bm25_top_k": r.bm25_top_k,
            "final_top_k": r.final_top_k,
            "rrf_k": r.rrf_k,
            "query_format": "Context + Question (options excluded)",
        }


def _matches(meta: dict, where: dict) -> bool:
    """BM25 端的簡易 metadata filter（支援 {key: value} 與 {key: {"$eq"/"$ne"/"$in": ...}}）。"""
    for key, cond in where.items():
        value = meta.get(key)
        if isinstance(cond, dict):
            for op, arg in cond.items():
                if op == "$eq" and value != arg:
                    return False
                if op == "$ne" and value == arg:
                    return False
                if op == "$in" and value not in arg:
                    return False
                if op not in ("$eq", "$ne", "$in"):
                    raise ValueError(f"Unsupported filter operator for BM25: {op}")
        elif value != cond:
            return False
    return True
