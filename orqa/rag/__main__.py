"""
RAG 獨立執行入口（不需要 LLM）：

    python -m orqa.rag stats    --config config/experiment.yaml      # chunk 統計（不呼叫 API）
    python -m orqa.rag build    --config config/experiment.yaml      # 建立 / 續建 vector index（Voyage）
    python -m orqa.rag status   --config config/experiment.yaml      # 檢查 index 狀態
    python -m orqa.rag list     --config config/experiment.yaml      # 列出所有 Chroma collections
    python -m orqa.rag retrieve --config config/experiment.yaml --limit 5
    python -m orqa.rag retrieve --config config/experiment.yaml --query "What is a binary variable?"
    python -m orqa.rag retrieve --config config/experiment.yaml --limit 5 --mode bm25

retrieve 的結果寫入 results/retrieval_runs/<timestamp>_<mode>.jsonl，方便分析 Retrieval Quality。
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from orqa.config import load_config
from orqa.dataset import load_dataset
from orqa.rag.pipeline import RAGPipeline, build_retrieval_query


def _pipeline(args) -> RAGPipeline:
    cfg = load_config(args.config)
    rag = cfg.rag
    if getattr(args, "mode", None):
        rag = dataclasses.replace(rag, retrieval=dataclasses.replace(rag.retrieval, mode=args.mode))
    return RAGPipeline(rag), cfg


def cmd_stats(args) -> None:
    p, _ = _pipeline(args)
    by_doc = collections.defaultdict(list)
    for c in p.chunks:
        by_doc[(c.document_name, c.chunk_strategy)].append(len(c.content))
    print(f"collection : {p.collection_name}")
    print(f"index_key  : {p.index_key}")
    for (name, strategy), lengths in by_doc.items():
        print(f"{name:<22} {strategy:<14} chunks={len(lengths):>5}  "
              f"chars mean={statistics.mean(lengths):7.1f} max={max(lengths)} min={min(lengths)}")
    cases = {c.case_id for c in p.chunks if c.case_id}
    print(f"total chunks = {len(p.chunks)}, distinct CASE ids = {len(cases)}")


def cmd_build(args) -> None:
    p, _ = _pipeline(args)
    print(f"Building index {p.collection_name} ({len(p.chunks)} chunks, model={p.settings.embedding.model})")
    result = p.build_index(force=args.force,
                           progress=lambda d, t: print(f"  embedded {d}/{t}", flush=True))
    print(json.dumps({k: v for k, v in result.items() if k != "metadata"}, indent=2, ensure_ascii=False))


def cmd_status(args) -> None:
    p, _ = _pipeline(args)
    print(json.dumps(p.index_status(), indent=2, ensure_ascii=False))


def cmd_list(args) -> None:
    p, _ = _pipeline(args)
    for c in p.store.list_collections():
        meta = c["metadata"]
        flag = "  <- current config" if c["name"] == p.collection_name else ""
        print(f"{c['name']}  count={c['count']}  status={meta.get('status')}  "
              f"model={meta.get('embedding_model')}{flag}")


def cmd_retrieve(args) -> None:
    p, cfg = _pipeline(args)
    if args.query:
        items = [(None, None, args.query)]
    else:
        ds = load_dataset(args.dataset or cfg.dataset.path).head(args.limit)
        items = [(q.index, q.question_type, build_retrieval_query(q)) for q in ds]

    out = Path(args.output) if args.output else (
        Path(cfg.experiment.output_dir) / "retrieval_runs"
        / f"{datetime.now():%Y%m%d_%H%M%S}_{p.settings.retrieval.mode}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for index, qtype, query in items:
            result = p.retrieve(query)
            rec = {"index": index, "question_type": qtype, **result.to_record()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"\n=== index={index} mode={result.mode} dense={result.dense_candidates} "
                  f"bm25={result.bm25_candidates} -> {len(result.chunks)} chunks")
            for c in result.chunks:
                sim = f"{c.dense_similarity:.3f}" if c.dense_similarity is not None else "  -  "
                bm = f"{c.bm25_score:.2f}" if c.bm25_score is not None else " - "
                print(f"  #{c.rank} {c.chunk_id:<28} score={c.score:.5f} dense_rank={c.dense_rank} "
                      f"sim={sim} bm25_rank={c.bm25_rank} bm25={bm}")
    print(f"\nSaved -> {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m orqa.rag", description="ORQA RAG utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stats", "build", "status", "list", "retrieve"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="config/experiment.yaml")
        if name == "build":
            sp.add_argument("--force", action="store_true",
                            help="刪除目前設定對應的 collection 後重建（其他 collection 不受影響）")
        if name == "retrieve":
            sp.add_argument("--limit", type=int, default=5)
            sp.add_argument("--query")
            sp.add_argument("--dataset")
            sp.add_argument("--output")
            sp.add_argument("--mode", choices=["dense", "bm25", "hybrid"])
    args = parser.parse_args(argv)
    {"stats": cmd_stats, "build": cmd_build, "status": cmd_status,
     "list": cmd_list, "retrieve": cmd_retrieve}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
