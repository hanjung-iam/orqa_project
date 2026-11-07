"""
Chunking 測試，包含 golden test：
直接執行 legacy/rag_colab_reference.py 中的舊版 chunking 函式，確認新版 chunk 文字逐字相同。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from orqa.rag.chunking import chunk_document, split_content

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "legacy" / "rag_colab_reference.py"
KB = ROOT / "data" / "knowledge_base"


def _legacy_namespace() -> dict:
    """擷取 Colab 匯出檔中 'CHUNK_SIZE = 2048' 到 'chunks = build_hybrid_chunks(' 之間的純 Python 程式碼並執行。"""
    src = LEGACY.read_text(encoding="utf-8")
    start = src.index("CHUNK_SIZE = 2048")
    end = src.index("chunks = build_hybrid_chunks(")
    ns: dict = {"re": re}
    exec(compile(src[start:end], str(LEGACY), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def legacy():
    return _legacy_namespace()


SAMPLE_MD = """Intro text before headings.

# Chapter 1
Short section.

## 1.1 Sub
""" + ("Paragraph " + "x" * 120 + "\n\n") * 30 + """
## 1.2 Long paragraph
""" + "y" * 5000 + """

# Chapter 2
### Deep
Final words.
"""

SAMPLE_CASES = """# Manifest
Preamble that is not part of any case.

## CASE small_001
**Model class:** LP

### Scenario
Tiny.

## CASE big_002
**Model class:** MILP
**Tags:** a | b

### Scenario
""" + ("Scenario text. " * 120) + """

### Sets
""" + ("- `S` set\n" * 40) + """

### Constraints
""" + ("z" * 4000) + """

### Objective
Minimize cost.
"""


def _legacy_contents(legacy, name, content, chunk_size=2048, stride=256):
    chunks = legacy["build_hybrid_chunks"]([{"document_name": name, "content": content}], chunk_size, stride)
    return chunks


@pytest.mark.parametrize("chunk_size,stride", [(2048, 256), (500, 50)])
def test_heading_golden_synthetic(legacy, chunk_size, stride):
    old = _legacy_contents(legacy, "version_0.md", SAMPLE_MD, chunk_size, stride)
    new = chunk_document("version_0.md", SAMPLE_MD, "heading_based", chunk_size, stride)
    assert [c["content"] for c in old] == [c.content for c in new]
    assert [c["chunk_heading"] for c in old] == [c.heading_path for c in new]
    assert any(c.heading_path == "Chapter 1 > 1.1 Sub" for c in new)
    assert all(len(c.content) <= chunk_size for c in new)


def test_case_golden_synthetic(legacy):
    # 舊版 build_case_chunk 長 section 切分時使用全域 STRIDE（=256），因此只比對 stride=256
    old = _legacy_contents(legacy, "StructuredOR.md", SAMPLE_CASES)
    new = chunk_document("StructuredOR.md", SAMPLE_CASES, "case_aware")
    assert [c["content"] for c in old] == [c.content for c in new]
    assert [c["case_id"] for c in old] == [c.case_id for c in new]
    assert [list(c["section_names"]) for c in old] == [list(c.section_names) for c in new]
    assert new[0].section_names == ("Full Case",) and new[0].case_id == "small_001"
    assert all(c.content.startswith("## CASE ") for c in new)          # 拆分後仍保留 CASE ID
    assert all(c.chunk_strategy == "case_aware" for c in new)          # 舊版短 case 誤用 strategy 欄位
    assert "Preamble" not in "".join(c.content for c in new)           # 與舊版相同：第一個 CASE 前的內容不納入
    big = [c for c in new if c.case_id == "big_002"]
    assert [c.chunk_index_in_case for c in big] == list(range(len(big)))
    assert len(big) > 1


@pytest.mark.skipif(not (KB / "StructuredOR.md").exists() or not (KB / "version_0.md").exists(),
                    reason="knowledge base documents not present in data/knowledge_base")
def test_golden_real_knowledge_base(legacy):
    from orqa.config import DocumentSpec
    from orqa.rag.documents import load_documents

    docs = load_documents([DocumentSpec(str(KB / "StructuredOR.md"), "case_aware"),
                           DocumentSpec(str(KB / "version_0.md"), "heading_based")])
    old = legacy["build_hybrid_chunks"]([{"document_name": d.name, "content": d.content} for d in docs], 2048, 256)
    new = [c for d in docs for c in chunk_document(d.name, d.content, d.strategy, 2048, 256)]
    assert len(old) == len(new)
    assert [c["content"] for c in old] == [c.content for c in new]
    assert {c.chunk_strategy for c in new} == {"case_aware", "heading_based"}
    assert len({c.chunk_id for c in new}) == len(new)


def test_split_content_overlap():
    para = "".join(chr(65 + i % 26) for i in range(1000))
    parts = split_content(para, chunk_size=300, stride=50)
    assert all(len(p) <= 300 for p in parts)
    for a, b in zip(parts, parts[1:]):
        assert a[-50:] == b[:50]          # 相鄰片段重疊 stride 字元
    assert parts[0] + "".join(p[50:] for p in parts[1:]) == para


def test_no_heading_document_does_not_crash():
    chunks = chunk_document("plain.md", "just text\n\nmore text", "heading_based")
    assert len(chunks) == 1 and chunks[0].heading_path == "Document Intro"


def test_metadata_is_chroma_safe():
    c = chunk_document("StructuredOR.md", SAMPLE_CASES, "case_aware")[1]
    meta = c.metadata(source_version="abc", embedding_model="voyage-4-large")
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
    assert json.loads(meta["section_names"]) == list(c.section_names)
    assert meta["case_id"] == "big_002" and meta["embedding_model"] == "voyage-4-large"
