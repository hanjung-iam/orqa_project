"""
orqa/config.py

YAML 設定檔載入與驗證。

- 所有實驗參數都來自 YAML（預設 config/experiment.yaml），不再從 .env 讀取實驗設定。
- .env 只用來提供 API Key（ANTHROPIC_API_KEY、VOYAGE_API_KEY、HF_TOKEN）。
- 設定物件為 frozen dataclass；CLI 覆寫透過 apply_overrides() 產生新物件。
"""

from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

METHODS = ("basic", "skill", "rag", "skill_rag")
PROVIDERS = ("anthropic", "transformers")
QUANTIZATIONS = ("none", "4bit", "8bit")
CHUNK_STRATEGIES = ("heading_based", "case_aware")
RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


class ConfigError(ValueError):
    """設定檔內容錯誤。"""


# ============================================================
# Sections
# ============================================================

@dataclass(frozen=True)
class ExperimentSection:
    name: str = "orqa_experiment"
    method: str = "basic"                  # basic | skill | rag | skill_rag
    output_dir: str = "results"
    experiment_id: Optional[str] = None    # None → 自動產生
    limit: Optional[int] = None            # 只跑前 N 題（smoke test）
    max_consecutive_errors: int = 5        # 連續失敗幾題後中止

    @property
    def use_skill(self) -> bool:
        return self.method in ("skill", "skill_rag")

    @property
    def use_rag(self) -> bool:
        return self.method in ("rag", "skill_rag")


@dataclass(frozen=True)
class DatasetSection:
    name: str = "ORQA_test"
    path: str = "dataset/ORQA_test.jsonl"


@dataclass(frozen=True)
class SkillSection:
    path: Optional[str] = None
    source_type: Optional[str] = None      # zip | directory | None(自動判斷)


@dataclass(frozen=True)
class PromptSection:
    version: str = "v2"


@dataclass(frozen=True)
class AnthropicSection:
    model_id: str = "claude-sonnet-5-5"
    max_tokens: int = 2048
    timeout: float = 120.0
    max_retries: int = 5
    prompt_caching: bool = True
    cache_ttl: Optional[str] = "1h"        # None → 預設 5 分鐘
    # 直接傳給 messages.create 的額外參數（例如 output_config: {effort: low}）。
    # 注意：API 版本 2026-10-21 起 temperature/top_p/top_k 一律回 400，不可放在這裡。
    extra_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TransformersSection:
    model_id: str = "Qwen/Qwen3.5-9B"
    revision: Optional[str] = None
    quantization: str = "4bit"             # none | 4bit | 8bit
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    dtype: str = "bfloat16"
    device_map: Optional[str] = "auto"     # None → 不使用 device_map，整個模型放到 device
    device: str = "cuda"                   # device_map 為 None 時使用
    max_memory: Optional[dict] = None      # 例如 {0: "11GiB", "cpu": "32GiB"}
    auto_model_class: str = "causal_lm"    # causal_lm | image_text_to_text
    attn_implementation: Optional[str] = None
    trust_remote_code: bool = False
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    batch_size: int = 1
    enable_thinking: bool = False
    empty_cache_every_batch: bool = False


@dataclass(frozen=True)
class LLMSection:
    provider: str = "anthropic"
    anthropic: AnthropicSection = field(default_factory=AnthropicSection)
    transformers: TransformersSection = field(default_factory=TransformersSection)

    @property
    def active(self) -> AnthropicSection | TransformersSection:
        return getattr(self, self.provider)


@dataclass(frozen=True)
class PostAnalysisSection:
    enabled: bool = True
    provider: str = "anthropic"
    model_id: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1024
    timeout: float = 120.0
    max_retries: int = 5


@dataclass(frozen=True)
class DocumentSpec:
    path: str
    strategy: str = "heading_based"        # heading_based | case_aware


@dataclass(frozen=True)
class EmbeddingSection:
    provider: str = "voyage"
    model: str = "voyage-4-large"
    batch_size: int = 128
    max_retries: int = 5
    output_dimension: Optional[int] = None


@dataclass(frozen=True)
class VectorStoreSection:
    provider: str = "chromadb"
    persist_directory: str = "data/vector_db"
    collection_prefix: str = "orqa_kb"


@dataclass(frozen=True)
class ChunkingSection:
    chunk_size: int = 2048
    stride: int = 256


@dataclass(frozen=True)
class RetrievalSection:
    mode: str = "hybrid"                   # dense | bm25 | hybrid
    similarity_threshold: float = 0.5      # 只作用於 Dense（cosine similarity）
    dense_top_k: int = 15
    bm25_top_k: int = 15
    final_top_k: int = 5
    rrf_k: int = 60


@dataclass(frozen=True)
class RAGSection:
    documents: tuple[DocumentSpec, ...] = (
        DocumentSpec("data/knowledge_base/StructuredOR.md", "case_aware"),
        DocumentSpec("data/knowledge_base/version_0.md", "heading_based"),
    )
    embedding: EmbeddingSection = field(default_factory=EmbeddingSection)
    vector_store: VectorStoreSection = field(default_factory=VectorStoreSection)
    chunking: ChunkingSection = field(default_factory=ChunkingSection)
    retrieval: RetrievalSection = field(default_factory=RetrievalSection)


@dataclass(frozen=True)
class Config:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    dataset: DatasetSection = field(default_factory=DatasetSection)
    skill: SkillSection = field(default_factory=SkillSection)
    prompt: PromptSection = field(default_factory=PromptSection)
    llm: LLMSection = field(default_factory=LLMSection)
    post_analysis: PostAnalysisSection = field(default_factory=PostAnalysisSection)
    rag: RAGSection = field(default_factory=RAGSection)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))


# ============================================================
# dict → dataclass
# ============================================================

def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _build(cls: type, data: Any, where: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"[{where}] 必須是 mapping，實際為 {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"[{where}] 未知的設定欄位: {sorted(unknown)}")

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _convert(hints[f.name], data[f.name], f"{where}.{f.name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"[{where}] {exc}") from exc


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        return _convert(non_none[0], value, where)

    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)

    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"[{where}] 必須是 list")
        item_tp = args[0] if args else Any
        return tuple(_convert(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value))

    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp in (int, float, str, bool) and value is not None and not isinstance(value, tp):
        raise ConfigError(f"[{where}] 型別應為 {tp.__name__}，實際為 {type(value).__name__}: {value!r}")
    if tp is int and isinstance(value, bool):
        raise ConfigError(f"[{where}] 型別應為 int，實際為 bool")
    return value


# ============================================================
# Public API
# ============================================================

def load_config(path: str | Path | None = None, *, validate: bool = True) -> Config:
    """讀取 YAML 設定檔。path 為 None 時回傳全部預設值。"""
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    config = _build(Config, data, "config")
    if validate:
        validate_config(config)
    return config


def config_from_dict(data: dict[str, Any], *, validate: bool = True) -> Config:
    """由 experiment.json 中的 config snapshot 重建設定（resume 用）。"""
    config = _build(Config, data, "config")
    if validate:
        validate_config(config)
    return config


def apply_overrides(config: Config, *, validate: bool = True, **overrides: Any) -> Config:
    """
    以 CLI 參數覆寫設定。支援的 key：
    method, provider, model_id, limit, experiment_id, dataset_path, skill_path
    （值為 None 的 key 會被忽略）
    """
    exp = config.experiment
    llm = config.llm
    dataset = config.dataset
    skill = config.skill

    if overrides.get("method") is not None:
        exp = dataclasses.replace(exp, method=overrides["method"])
    if overrides.get("limit") is not None:
        exp = dataclasses.replace(exp, limit=int(overrides["limit"]))
    if overrides.get("experiment_id") is not None:
        exp = dataclasses.replace(exp, experiment_id=overrides["experiment_id"])
    if overrides.get("provider") is not None:
        llm = dataclasses.replace(llm, provider=overrides["provider"])
    if overrides.get("model_id") is not None:
        section = dataclasses.replace(llm.active, model_id=overrides["model_id"])
        llm = dataclasses.replace(llm, **{llm.provider: section})
    if overrides.get("dataset_path") is not None:
        dataset = dataclasses.replace(dataset, path=overrides["dataset_path"])
    if overrides.get("skill_path") is not None:
        skill = dataclasses.replace(skill, path=overrides["skill_path"])

    new = dataclasses.replace(config, experiment=exp, llm=llm, dataset=dataset, skill=skill)
    if validate:
        validate_config(new)
    return new


def validate_config(config: Config, *, check_paths: bool = True) -> None:
    exp = config.experiment
    if exp.method not in METHODS:
        raise ConfigError(f"experiment.method 必須是 {METHODS}，實際為 {exp.method!r}")
    if exp.limit is not None and exp.limit <= 0:
        raise ConfigError("experiment.limit 必須 > 0")
    if exp.max_consecutive_errors <= 0:
        raise ConfigError("experiment.max_consecutive_errors 必須 > 0")

    if check_paths and not Path(config.dataset.path).exists():
        raise ConfigError(f"dataset.path 不存在: {config.dataset.path}")

    if exp.use_skill:
        if not config.skill.path:
            raise ConfigError(f"method={exp.method} 需要設定 skill.path")
        if check_paths and not Path(config.skill.path).exists():
            raise ConfigError(f"skill.path 不存在: {config.skill.path}")
    if config.skill.source_type not in (None, "zip", "directory"):
        raise ConfigError("skill.source_type 必須是 zip / directory / null")

    llm = config.llm
    if llm.provider not in PROVIDERS:
        raise ConfigError(f"llm.provider 必須是 {PROVIDERS}，實際為 {llm.provider!r}")

    a = llm.anthropic
    if a.max_tokens <= 0:
        raise ConfigError("llm.anthropic.max_tokens 必須 > 0")
    banned = {"temperature", "top_p", "top_k"} & set(a.extra_params)
    if banned:
        raise ConfigError(
            f"llm.anthropic.extra_params 不可包含 {sorted(banned)}："
            "Anthropic API 版本 2026-10-21 起，傳送任何 sampling 參數都會回 400。"
        )

    t = llm.transformers
    if t.quantization not in QUANTIZATIONS:
        raise ConfigError(f"llm.transformers.quantization 必須是 {QUANTIZATIONS}")
    if t.auto_model_class not in ("causal_lm", "image_text_to_text"):
        raise ConfigError("llm.transformers.auto_model_class 必須是 causal_lm / image_text_to_text")
    if t.max_new_tokens <= 0 or t.batch_size <= 0:
        raise ConfigError("llm.transformers.max_new_tokens / batch_size 必須 > 0")
    if not t.do_sample and any(v is not None for v in (t.temperature, t.top_p, t.top_k)):
        raise ConfigError("llm.transformers: do_sample=false 時不可設定 temperature/top_p/top_k")

    pa = config.post_analysis
    if pa.enabled and pa.provider != "anthropic":
        raise ConfigError("post_analysis.provider 目前只支援 anthropic")

    rag = config.rag
    if not rag.documents:
        raise ConfigError("rag.documents 不可為空")
    for doc in rag.documents:
        if doc.strategy not in CHUNK_STRATEGIES:
            raise ConfigError(f"rag.documents.strategy 必須是 {CHUNK_STRATEGIES}: {doc}")
    if rag.embedding.provider != "voyage":
        raise ConfigError("rag.embedding.provider 目前只支援 voyage")
    if rag.vector_store.provider != "chromadb":
        raise ConfigError("rag.vector_store.provider 目前只支援 chromadb")
    c = rag.chunking
    if c.chunk_size <= 0 or not (0 <= c.stride < c.chunk_size):
        raise ConfigError("rag.chunking: 需滿足 chunk_size > 0 且 0 <= stride < chunk_size")
    r = rag.retrieval
    if r.mode not in RETRIEVAL_MODES:
        raise ConfigError(f"rag.retrieval.mode 必須是 {RETRIEVAL_MODES}")
    if not (0.0 <= r.similarity_threshold <= 1.0):
        raise ConfigError("rag.retrieval.similarity_threshold 必須介於 0 與 1")
    for name in ("dense_top_k", "bm25_top_k", "final_top_k", "rrf_k"):
        if getattr(r, name) <= 0:
            raise ConfigError(f"rag.retrieval.{name} 必須 > 0")


# ============================================================
# Secrets
# ============================================================

_DOTENV_LOADED = False


def get_secret(name: str, *, required: bool = True) -> Optional[str]:
    """從環境變數（含 .env）讀取 API Key。不會寫入任何檔案或 log。"""
    global _DOTENV_LOADED
    if not _DOTENV_LOADED:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass
        _DOTENV_LOADED = True

    value = os.getenv(name)
    if value is not None:
        value = value.strip()
    if not value:
        if required:
            raise ConfigError(f"環境變數 {name} 未設定（請寫入 .env，參考 .env.example）")
        return None
    return value
