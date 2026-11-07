"""
orqa/rag/documents.py

知識庫文件載入與版本識別（SHA256）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from orqa.config import DocumentSpec


@dataclass(frozen=True)
class KnowledgeDocument:
    name: str           # 檔名，例如 StructuredOR.md
    path: str
    strategy: str
    content: str
    sha256: str

    @property
    def source_version(self) -> str:
        return self.sha256[:16]


def load_documents(specs: Sequence[DocumentSpec]) -> list[KnowledgeDocument]:
    docs = []
    names = set()
    for spec in specs:
        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(
                f"Knowledge base document not found: {path} "
                "(請將文件放在 data/knowledge_base/，或修改 config 的 rag.documents)"
            )
        raw = path.read_bytes()
        # 統一換行，讓 chunk 內容與 hash 不受 Windows CRLF 影響
        content = raw.decode("utf-8-sig").replace("\r\n", "\n")
        if path.name in names:
            raise ValueError(f"Duplicate document name in rag.documents: {path.name}")
        names.add(path.name)
        docs.append(KnowledgeDocument(
            name=path.name,
            path=path.as_posix(),
            strategy=spec.strategy,
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ))
    return docs
