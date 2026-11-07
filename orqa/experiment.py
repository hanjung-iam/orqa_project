"""
orqa/experiment.py

Experiment Management：每次實驗一個獨立目錄 results/<experiment_id>/。

- 新實驗一律建立新的目錄；目錄已存在時拒絕建立（絕不覆寫舊結果）。
- experiment.json 保存完整設定 snapshot、資料集 / Skill / Prompt / RAG index 的指紋、
  provider 資訊、狀態與每次執行的紀錄。
- Resume 時使用 experiment.json 中的設定 snapshot（而不是目前的 YAML），
  並檢查資料集、Skill、Prompt、RAG index 指紋是否一致，不一致就拒絕續跑。
"""

from __future__ import annotations

import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from orqa import __version__
from orqa.config import Config, config_from_dict
from orqa.storage import EXPERIMENT, ExperimentStorage, read_json, write_json_atomic

STATUS_CREATED = "created"
STATUS_RUNNING = "running_inference"
STATUS_INFERENCE_DONE = "inference_complete"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"
STATUS_POST_DONE = "post_analysis_complete"
STATUS_EVALUATED = "evaluated"


class ExperimentError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(text: str) -> str:
    text = text.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:40]


def make_experiment_id(config: Config) -> str:
    model = config.llm.active.model_id
    return f"{datetime.now():%Y%m%d_%H%M%S}_{config.experiment.method}_{_slug(model)}"


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "orqa", "config", "main.py"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


class Experiment:

    def __init__(self, directory: Path, meta: dict[str, Any]):
        self.dir = directory
        self.meta = meta
        self.storage = ExperimentStorage(directory)

    @property
    def id(self) -> str:
        return self.meta["experiment_id"]

    @property
    def config(self) -> Config:
        return config_from_dict(self.meta["config"], validate=False)

    # -------------------------------------------------------------- create

    @classmethod
    def create(cls, config: Config, *, dataset, skill, prompt_info: dict, provider_info: dict,
               rag_info: Optional[dict], experiment_id: Optional[str] = None) -> "Experiment":
        experiment_id = experiment_id or make_experiment_id(config)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", experiment_id):
            raise ExperimentError(f"Invalid experiment_id: {experiment_id!r}")
        directory = Path(config.experiment.output_dir) / experiment_id
        if directory.exists():
            raise ExperimentError(
                f"{directory} already exists. Use --resume {experiment_id} to continue it; "
                "new experiments never overwrite existing results."
            )
        directory.mkdir(parents=True, exist_ok=False)

        exp = config.experiment
        rag = config.rag.retrieval
        meta: dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp": _now(),
            "experiment_name": exp.name,
            "experiment_status": STATUS_CREATED,
            "dataset_name": config.dataset.name,
            "provider": config.llm.provider,
            "model_id": config.llm.active.model_id,
            "quantization": provider_info.get("quantization"),
            "inference_mode": exp.method,
            "skill_enabled": exp.use_skill,
            "RAG_enabled": exp.use_rag,
            "embedding_model": config.rag.embedding.model if exp.use_rag else None,
            "similarity_threshold": rag.similarity_threshold if exp.use_rag else None,
            "dense_top_k": rag.dense_top_k if exp.use_rag else None,
            "bm25_top_k": rag.bm25_top_k if exp.use_rag else None,
            "rrf_k": rag.rrf_k if exp.use_rag else None,
            "final_top_k": rag.final_top_k if exp.use_rag else None,
            "retrieval_mode": rag.mode if exp.use_rag else None,
            "generation_parameters": provider_info.get("generation_parameters"),
            "post_analysis_model": config.post_analysis.model_id if config.post_analysis.enabled else None,
            "dataset": {
                "path": str(dataset.path),
                "sha256": dataset.sha256,
                "total_questions": len(dataset),
                "limit": exp.limit,
                "evaluated_indices": [min(dataset.indices), max(dataset.indices)] if len(dataset) else None,
            },
            "skill": skill.metadata() if skill is not None else None,
            "prompt": prompt_info,
            "provider_info": provider_info,
            "rag": rag_info,
            "environment": {
                "orqa_version": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git": _git_info(),
            },
            "config": config.to_dict(),
            "runs": [],
            "files": {},
        }
        write_json_atomic(directory / EXPERIMENT, meta)
        return cls(directory, meta)

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, output_dir: str | Path, experiment_id: str) -> "Experiment":
        directory = Path(output_dir) / experiment_id
        path = directory / EXPERIMENT
        if not path.exists():
            raise ExperimentError(f"Experiment not found: {path}")
        meta = read_json(path)
        if "config" not in meta:
            raise ExperimentError(f"{path} is not a v2 experiment (no config snapshot); legacy results are read-only")
        return cls(directory, meta)

    def verify(self, *, dataset, skill, prompt_sha256: str, rag_index_key: Optional[str]) -> None:
        """resume 前檢查輸入是否與建立實驗時相同。"""
        problems = []
        if dataset.sha256 != self.meta["dataset"]["sha256"]:
            problems.append("dataset file content changed (question index alignment is no longer guaranteed)")
        old_skill = (self.meta.get("skill") or {}).get("sha256")
        new_skill = skill.sha256 if skill is not None else None
        if old_skill != new_skill:
            problems.append("skill instructions changed")
        if prompt_sha256 != self.meta["prompt"]["system_sha256"]:
            problems.append("system prompt changed")
        old_rag = (self.meta.get("rag") or {}).get("index_key")
        if old_rag != rag_index_key:
            problems.append("RAG index (documents / chunking / embedding) changed")
        if problems:
            raise ExperimentError(f"Cannot resume {self.id}: " + "; ".join(problems))

    # --------------------------------------------------------------- update

    def save(self) -> None:
        write_json_atomic(self.dir / EXPERIMENT, self.meta)

    def set_status(self, status: str) -> None:
        self.meta["experiment_status"] = status
        self.meta["updated_at"] = _now()
        self.save()

    def start_run(self, stage: str, **info: Any) -> dict[str, Any]:
        run = {"stage": stage, "started_at": _now(), **info}
        self.meta.setdefault("runs", []).append(run)
        self.save()
        return run

    def finish_run(self, run: dict[str, Any], **info: Any) -> None:
        run.update({"finished_at": _now(), **info})
        self.save()

    def record_file(self, name: str, **info: Any) -> None:
        self.meta.setdefault("files", {})[name] = {"updated_at": _now(), **info}
        self.save()
