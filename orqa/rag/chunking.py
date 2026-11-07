"""
orqa/rag/chunking.py

Hybrid Chunking —— 從 legacy/rag_colab_reference.py 移植，chunk 文字內容與舊版逐字相同
（由 tests/unit/test_rag_chunking.py 的 golden test 驗證）。

A. heading_based（一般 Markdown，例如 version_0.md）
   1. 解析 Markdown heading（# ~ ######），建立 heading hierarchy
   2. 每個 section 的 heading_path = "H1 > H2 > ..."
   3. section 內容 <= chunk_size 直接成為一個 chunk；否則依空白行段落累積，
      單一段落過長時以固定長度切分，相鄰片段重疊 stride 字元

B. case_aware（StructuredOR.md）
   1. 以 "## CASE <case_id>" 作為 case 邊界（第一個 CASE 之前的內容不納入，與舊版相同）
   2. 整個 case <= chunk_size → 一個 chunk（section_names = ["Full Case"]）
   3. 否則解析 "###" section，優先把同一 case 的多個 section 放進同一個 chunk，
      超過 chunk_size 才拆分；每個 chunk 都以 "## CASE <id>" 開頭，保留 case id

與舊版的差異（只有 metadata，不影響 chunk 文字）：
- 統一使用 chunk_strategy 欄位（舊版短 case 誤用 strategy）
- heading chunk 的 chunk_heading 改名為 heading_path
- chunk_id 改為「文件名稱 + 文件內序號」，不再是跨文件的全域序號
- 文件沒有任何 heading 時不會 crash（舊版會 IndexError），整份視為 "Document Intro"
- 切分策略由設定檔指定，不再依檔名判斷
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

CHUNKER_VERSION = "hybrid-v1"   # chunk 邏輯改變時要更新，會讓 vector index 重建

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_CASE = re.compile(r"^##\s+CASE\s+(.+?)\s*$", re.MULTILINE)
_CASE_SECTION = re.compile(r"^(###)\s+(.+?)\s*$", re.MULTILINE)
_CASE_HEADING = re.compile(r"^##\s+CASE\s+.+?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_name: str
    chunk_strategy: str            # heading_based | case_aware
    chunk_index: int               # 文件內序號
    content: str
    heading_path: str = ""
    section_index: int = -1
    case_id: str = ""
    chunk_index_in_case: int = -1
    section_names: tuple[str, ...] = field(default_factory=tuple)

    def metadata(self, **extra: Any) -> dict[str, Any]:
        """ChromaDB-safe metadata：只有 str / int / float / bool，list 轉成 JSON 字串。"""
        meta: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "chunk_strategy": self.chunk_strategy,
            "chunk_index": self.chunk_index,
            "heading_path": self.heading_path,
            "section_index": self.section_index,
            "case_id": self.case_id,
            "chunk_index_in_case": self.chunk_index_in_case,
            "section_names": json.dumps(list(self.section_names), ensure_ascii=False),
            "content_length": len(self.content),
        }
        for key, value in extra.items():
            if value is None:
                value = ""
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"metadata {key} must be a scalar, got {type(value).__name__}")
            meta[key] = value
        return meta


# ----------------------------------------------------------------- helpers

def split_content(content: str, chunk_size: int, stride: int) -> list[str]:
    """與舊版 split_content 完全相同。"""
    content = content.strip()
    if len(content) <= chunk_size:
        return [content]

    paragraphs = re.split(r"\n\s*\n", content)
    chunks: list[str] = []
    temp_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(temp_chunk) + len(paragraph) + 2 <= chunk_size:
            if temp_chunk:
                temp_chunk += "\n\n"
            temp_chunk += paragraph
        else:
            if temp_chunk:
                chunks.append(temp_chunk.strip())

            if len(paragraph) <= chunk_size:
                temp_chunk = paragraph
            else:
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunks.append(paragraph[start:end])
                    if end >= len(paragraph):
                        break
                    start = end - stride
                temp_chunk = ""

    if temp_chunk:
        chunks.append(temp_chunk.strip())
    return chunks


def heading_parsing(content: str) -> list[dict[str, str]]:
    headings = list(_HEADING.finditer(content))
    if not headings:
        stripped = content.strip()
        return [{"heading_path": "Document Intro", "content": stripped}] if stripped else []

    sections = []
    stack: list[dict[str, Any]] = []
    pre = content[: headings[0].start()].strip()
    if pre:
        sections.append({"heading_path": "Document Intro", "content": pre})

    for i, h in enumerate(headings):
        level = len(h.group(1))
        text = h.group(2).strip()
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        stack.append({"level": level, "text": text})

        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        body = content[h.end():end].strip()
        if body:
            sections.append({
                "heading_path": " > ".join(item["text"] for item in stack),
                "content": body,
            })
    return sections


def case_parsing(content: str) -> list[dict[str, str]]:
    matches = list(_CASE.finditer(content))
    cases = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        cases.append({"case_id": m.group(1).strip(), "content": content[m.start():end].strip()})
    return cases


def case_sec_parsing(case_content: str) -> list[dict[str, str]]:
    section_matches = list(_CASE_SECTION.finditer(case_content))
    if not section_matches:
        return [{"section_name": "Full Case", "content": case_content}]

    sections = []
    heading = _CASE_HEADING.match(case_content)
    meta_start = heading.end() if heading else 0
    metadata = case_content[meta_start:section_matches[0].start()].strip()
    if metadata:
        sections.append({"section_name": "Case Metadata", "content": metadata})

    for i, m in enumerate(section_matches):
        end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(case_content)
        body = case_content[m.end():end].strip()
        if body:
            sections.append({"section_name": m.group(2).strip(), "content": body})
    return sections


# --------------------------------------------------------------- strategies

def _heading_chunks(document_name: str, content: str, chunk_size: int, stride: int) -> list[dict]:
    out = []
    for section_index, section in enumerate(heading_parsing(content)):
        for part in split_content(section["content"], chunk_size, stride):
            out.append({
                "chunk_strategy": "heading_based",
                "section_index": section_index,
                "heading_path": section["heading_path"],
                "content": part,
            })
    return out


def _case_chunks(document_name: str, content: str, chunk_size: int, stride: int) -> list[dict]:
    out = []
    for case in case_parsing(content):
        case_id, case_content = case["case_id"], case["content"]

        if len(case_content) <= chunk_size:
            out.append({"case_id": case_id, "chunk_index_in_case": 0,
                        "section_names": ("Full Case",), "content": case_content})
            continue

        prefix = f"## CASE {case_id}\n\n"
        current = f"## CASE {case_id}\n"
        names: list[str] = []
        idx = 0
        for section in case_sec_parsing(case_content):
            name = section["section_name"]
            text = f"### {name}\n{section['content']}"

            if len(current) + len(text) + 2 <= chunk_size:
                current += "\n\n" + text
                names.append(name)
                continue

            if names:
                out.append({"case_id": case_id, "chunk_index_in_case": idx,
                            "section_names": tuple(names), "content": current.strip()})
                idx += 1

            if len(text) <= chunk_size:
                current = prefix + text
                names = [name]
            else:
                for part in split_content(text, chunk_size=chunk_size - len(prefix), stride=stride):
                    out.append({"case_id": case_id, "chunk_index_in_case": idx,
                                "section_names": (name,), "content": prefix + part})
                    idx += 1
                current = f"## CASE {case_id}\n"
                names = []

        if names:
            out.append({"case_id": case_id, "chunk_index_in_case": idx,
                        "section_names": tuple(names), "content": current.strip()})

    for item in out:
        item["chunk_strategy"] = "case_aware"
    return out


STRATEGIES = {
    "heading_based": _heading_chunks,
    "case_aware": _case_chunks,
}


def chunk_document(document_name: str, content: str, strategy: str,
                   chunk_size: int = 2048, stride: int = 256) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunk strategy: {strategy}")
    raw = STRATEGIES[strategy](document_name, content, chunk_size, stride)
    return [
        Chunk(chunk_id=f"{document_name}::{i:05d}", document_name=document_name, chunk_index=i, **item)
        for i, item in enumerate(raw)
    ]
