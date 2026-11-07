"""
orqa/rag/vector_store.py

本機 ChromaDB（PersistentClient）封裝。

- Collection 一律使用 cosine distance（configuration={"hnsw": {"space": "cosine"}}），
  ChromaDB 回傳的是 cosine distance = 1 - cosine similarity。
- 不使用 Chroma 內建的 embedding function，embedding 一律由 Voyage 產生後傳入。
- Collection metadata 記錄 index 狀態（building / complete）與版本資訊。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

UPSERT_BATCH = 512


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


class ChromaVectorStore:

    def __init__(self, persist_directory: str | Path, *, client: Any = None):
        import chromadb
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = client or chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = None
        self.collection_name: Optional[str] = None

    # ------------------------------------------------------------ collections

    def list_collections(self) -> list[dict[str, Any]]:
        out = []
        for c in self.client.list_collections():
            col = self.client.get_collection(c.name if hasattr(c, "name") else c, embedding_function=None)
            out.append({"name": col.name, "count": col.count(), "metadata": dict(col.metadata or {})})
        return out

    def exists(self, name: str) -> bool:
        return any((c.name if hasattr(c, "name") else c) == name for c in self.client.list_collections())

    def open(self, name: str, *, create: bool = False, metadata: Optional[dict] = None) -> None:
        if create:
            col = self.client.get_or_create_collection(
                name=name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata=metadata or None,
                embedding_function=None,
            )
        else:
            if not self.exists(name):
                raise VectorStoreError(f"Chroma collection not found: {name}")
            col = self.client.get_collection(name, embedding_function=None)
        space = (col.configuration or {}).get("hnsw", {}).get("space")
        if space != "cosine":
            raise VectorStoreError(f"Collection {name} uses distance '{space}', expected 'cosine'")
        self._collection = col
        self.collection_name = name

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)
        if self.collection_name == name:
            self._collection = None
            self.collection_name = None

    @property
    def collection(self):
        if self._collection is None:
            raise VectorStoreError("No collection opened")
        return self._collection

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.collection.metadata or {})

    def update_metadata(self, **values: Any) -> None:
        meta = self.metadata
        meta.update(values)
        self.collection.modify(metadata=meta)

    # --------------------------------------------------------------- records

    def count(self) -> int:
        return self.collection.count()

    def ids(self) -> set[str]:
        return set(self.collection.get(include=[])["ids"])

    def upsert(self, ids: Sequence[str], embeddings: Sequence[Sequence[float]],
               documents: Sequence[str], metadatas: Sequence[dict]) -> None:
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise VectorStoreError("upsert: ids / embeddings / documents / metadatas length mismatch")
        for s in range(0, len(ids), UPSERT_BATCH):
            e = s + UPSERT_BATCH
            self.collection.upsert(ids=list(ids[s:e]), embeddings=[list(v) for v in embeddings[s:e]],
                                   documents=list(documents[s:e]), metadatas=list(metadatas[s:e]))

    def delete(self, *, ids: Optional[Iterable[str]] = None, where: Optional[dict] = None) -> None:
        if ids is None and where is None:
            raise VectorStoreError("delete requires ids or where (refusing to delete everything)")
        self.collection.delete(ids=list(ids) if ids is not None else None, where=where)

    def query(self, embedding: Sequence[float], n_results: int,
              where: Optional[dict] = None) -> list[VectorHit]:
        r = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            VectorHit(chunk_id=i, content=d, metadata=dict(m or {}), distance=float(dist))
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
