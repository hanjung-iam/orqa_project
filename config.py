from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Experiment Configuration
# ============================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """
    整個 ORQA 實驗的總設定。

    所有模組都應該從這個 Config 取得設定，
    不要在各個 .py 裡面重新定義 MODEL、MAX_TOKENS 等參數。
    """

    experiment: ExperimentSettings
    dataset: DatasetSettings
    skill: SkillSettings
    rag: RAGSettings
    llm: LLMSettings
    cache: CacheSettings
    repair: RepairSettings
    checkpoint: CheckpointSettings
    output: OutputSettings
    logging: LoggingSettings

    @property
    def method(self) -> str:
        """
        根據 use_skill / use_rag 自動決定目前實驗方法。
        """

        if self.experiment.use_skill and self.experiment.use_rag:
            return "skill_rag"

        if self.experiment.use_skill:
            return "skill"

        if self.experiment.use_rag:
            return "rag"

        return "basic"

@dataclass(frozen=True)
class ExperimentSettings:
    use_skill: bool = False
    use_rag: bool = False

@dataclass(frozen=True)
class DatasetSettings:
    path: Path
    data_type: str = "jsonl"
    split: str = "test"

@dataclass(frozen=True)
class SkillSettings:
    enabled: bool = False
    path: Optional[Path] = None
    # ZIP / skill directory
    source_type: str = "zip"

@dataclass(frozen=True)
class RAGSettings:
    """
    RAG 相關設定。

    注意：
    目前改成使用 RAGFlow，因此這裡不再放
    Voyage / Chroma / BM25 / RRF 的舊設定。

    之後 RAGFlow adapter 再負責實際 retrieval。
    """

    enabled: bool = False
    provider: str = "ragflow"

    # RAGFlow server
    host: Optional[str] = None
    api_key: Optional[str] = None

    # RAGFlow Knowledge Base / Dataset
    dataset_id: Optional[str] = None

    # Retrieval
    top_k: int = 5

    # 是否在結果中保存 retrieved chunks
    save_retrieved_chunks: bool = True

@dataclass(frozen=True)
class LLMSettings:
    provider: str = "anthropic"
    api_key: Optional[str] = None
    model: str = "claude-opus-5"
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: int = 120
    max_retries: int = 3

@dataclass(frozen=True)
class CacheSettings:
    enabled: bool = True

@dataclass(frozen=True)
class RepairSettings:
    enabled: bool = True
    provider: str = "anthropic"
    api_key: Optional[str] = None
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: int = 120
    max_retries: int = 3

@dataclass(frozen=True)
class CheckpointSettings:
    enabled: bool = True
    save_every: int = 1     # 每完成幾題儲存一次

@dataclass(frozen=True)
class OutputSettings:
    base_dir: Path = Path("results")
    # 如果 None，之後由程式自動產生 timestamp
    run_id: Optional[str] = None

@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return value.strip()


def _get_bool_env(name: str, default: bool) -> bool:

    value = _get_env(name)

    if value is None:
        return default

    value = value.lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    raise ValueError(
        f"Environment variable {name} must be true/false."
    )


def _get_int_env(name: str, default: int) -> int:

    value = _get_env(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be an integer."
        )


def _get_float_env(name: str, default: float) -> float:

    value = _get_env(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be a number."
        )


def create_config(
    *,
    use_skill: bool = False,
    use_rag: bool = False,
    dataset_path: Optional[str] = None,
    skill_path: Optional[str] = None,
) -> ExperimentConfig:
    """
    建立完整 ExperimentConfig。

    注意：
    use_skill / use_rag / dataset_path / skill_path
    目前可以由 main.py / CLI 傳入。

    API Key 等機密資訊則從 .env 讀取。
    """
    # Dataset

    if dataset_path is None:
        dataset_path = _get_env(
            "ORQA_DATASET_PATH",
            "data/ORQA_test.jsonl",
        )

    dataset_settings = DatasetSettings(
        path=Path(dataset_path),
        data_type=_get_env("ORQA_DATA_TYPE", "jsonl"),
        split=_get_env("ORQA_DATA_SPLIT", "test"),
    )

    # Skill
    
    if skill_path is None:
        skill_path = _get_env("SKILL_PATH")

    skill_settings = SkillSettings(
        enabled=use_skill,
        path=Path(skill_path) if skill_path else None,
        source_type=_get_env("SKILL_SOURCE_TYPE", "zip"),
    )

    # RAGFlow
    
    rag_settings = RAGSettings(
        enabled=use_rag,
        provider=_get_env("RAG_PROVIDER", "ragflow"),
        host=_get_env("RAGFLOW_HOST"),
        api_key=_get_env("RAGFLOW_API_KEY"),
        dataset_id=_get_env("RAGFLOW_DATASET_ID"),
        top_k=_get_int_env("RAG_TOP_K", 5),
        save_retrieved_chunks=_get_bool_env(
            "RAG_SAVE_RETRIEVED_CHUNKS",
            True,
        ),
    )

    # Main LLM

    llm_settings = LLMSettings(
        provider=_get_env("LLM_PROVIDER", "anthropic"),
        api_key=_get_env("ANTHROPIC_API_KEY"),
        model=_get_env("LLM_MODEL", "claude-opus-5"),
        max_tokens=_get_int_env("LLM_MAX_TOKENS", 2048),
        temperature=_get_float_env("LLM_TEMPERATURE", 0.0),
        timeout=_get_int_env("LLM_TIMEOUT", 120),
        max_retries=_get_int_env("LLM_MAX_RETRIES", 3),
    )

    # Cache
    
    cache_settings = CacheSettings(
        enabled=_get_bool_env(
            "CACHE_ENABLED",
            True,
        )
    )

    # Answer Repair
    
    repair_settings = RepairSettings(
        enabled=_get_bool_env(
            "REPAIR_ENABLED",
            True,
        ),
        provider=_get_env(
            "REPAIR_PROVIDER",
            "anthropic",
        ),
        api_key=_get_env(
            "ANTHROPIC_API_KEY",
        ),
        model=_get_env(
            "REPAIR_MODEL",
            "claude-haiku-4-5-20251001",
        ),
        max_tokens=_get_int_env(
            "REPAIR_MAX_TOKENS",
            2048,
        ),
        temperature=_get_float_env(
            "REPAIR_TEMPERATURE",
            0.0,
        ),
        timeout=_get_int_env(
            "REPAIR_TIMEOUT",
            120,
        ),
        max_retries=_get_int_env(
            "REPAIR_MAX_RETRIES",
            3,
        ),
    )

    # Checkpoint
    
    checkpoint_settings = CheckpointSettings(
        enabled=_get_bool_env(
            "CHECKPOINT_ENABLED",
            True,
        ),
        save_every=_get_int_env(
            "CHECKPOINT_SAVE_EVERY",
            1,
        ),
    )

    # Output
    
    output_settings = OutputSettings(
        base_dir=Path(
            _get_env(
                "OUTPUT_BASE_DIR",
                "results",
            )
        ),
        run_id=_get_env("RUN_ID"),
    )

    # Logging
    
    logging_settings = LoggingSettings(
        level=_get_env(
            "LOG_LEVEL",
            "INFO",
        )
    )

    # Experiment
    
    experiment_settings = ExperimentSettings(
        use_skill=use_skill,
        use_rag=use_rag,
    )

    # Final Config
    
    config = ExperimentConfig(
        experiment=experiment_settings,
        dataset=dataset_settings,
        skill=skill_settings,
        rag=rag_settings,
        llm=llm_settings,
        cache=cache_settings,
        repair=repair_settings,
        checkpoint=checkpoint_settings,
        output=output_settings,
        logging=logging_settings,
    )

    # Validate
   
    validate_config(config)

    return config


# Validation

def validate_config(config: ExperimentConfig) -> None:
    """
    驗證 ExperimentConfig 是否合理。

    如果設定錯誤，在真正開始執行 API / Dataset / RAG 前就直接報錯。
    """

    if config.dataset.data_type != "jsonl":
        raise ValueError(
            f"Unsupported dataset type: "
            f"{config.dataset.data_type}"
        )

    if not config.dataset.path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist: "
            f"{config.dataset.path}"
        )

    if config.experiment.use_skill:

        if not config.skill.enabled:
            raise ValueError(
                "use_skill=True but Skill is disabled."
            )

        if config.skill.path is None:
            raise ValueError(
                "Skill is enabled but SKILL_PATH is not set."
            )

        if not config.skill.path.exists():
            raise FileNotFoundError(
                f"Skill file/directory does not exist: "
                f"{config.skill.path}"
            )

        if config.skill.source_type not in {
            "zip",
            "directory",
        }:
            raise ValueError(
                "SKILL_SOURCE_TYPE must be "
                "'zip' or 'directory'."
            )

    if config.experiment.use_rag:

        if not config.rag.enabled:
            raise ValueError(
                "use_rag=True but RAG is disabled."
            )

        if config.rag.provider != "ragflow":
            raise ValueError(
                f"Unsupported RAG provider: "
                f"{config.rag.provider}"
            )

        if not config.rag.host:
            raise ValueError(
                "RAG is enabled but RAGFLOW_HOST is not set."
            )

        if not config.rag.api_key:
            raise ValueError(
                "RAG is enabled but RAGFLOW_API_KEY is not set."
            )

        if not config.rag.dataset_id:
            raise ValueError(
                "RAG is enabled but "
                "RAGFLOW_DATASET_ID is not set."
            )

        if config.rag.top_k <= 0:
            raise ValueError(
                "RAG_TOP_K must be greater than 0."
            )
        
    if config.llm.provider != "anthropic":
        raise ValueError(
            f"Unsupported LLM provider: "
            f"{config.llm.provider}"
        )

    if not config.llm.api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set."
        )

    if config.llm.max_tokens <= 0:
        raise ValueError(
            "LLM_MAX_TOKENS must be greater than 0."
        )

    if config.llm.temperature < 0:
        raise ValueError(
            "LLM_TEMPERATURE cannot be negative."
        )

    
    if config.repair.enabled:

        if not config.repair.api_key:
            raise ValueError(
                "Repair is enabled but "
                "ANTHROPIC_API_KEY is not set."
            )

        if config.repair.max_tokens <= 0:
            raise ValueError(
                "REPAIR_MAX_TOKENS must be greater than 0."
            )

    
    if config.checkpoint.save_every <= 0:
        raise ValueError(
            "CHECKPOINT_SAVE_EVERY must be greater than 0."
        )

    
    valid_log_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if config.logging.level.upper() not in valid_log_levels:
        raise ValueError(
            f"Invalid log level: "
            f"{config.logging.level}"
        )



def print_config(config: ExperimentConfig) -> None:

    print("=" * 60)
    print("ORQA Experiment Configuration")
    print("=" * 60)

    print(f"Method                : {config.method}")

    print("\n[Experiment]")
    print(f"Use Skill             : {config.experiment.use_skill}")
    print(f"Use RAG               : {config.experiment.use_rag}")

    print("\n[Dataset]")
    print(f"Path                  : {config.dataset.path}")
    print(f"Type                  : {config.dataset.data_type}")
    print(f"Split                 : {config.dataset.split}")

    print("\n[Skill]")
    print(f"Enabled               : {config.skill.enabled}")
    print(f"Path                  : {config.skill.path}")
    print(f"Source Type           : {config.skill.source_type}")

    print("\n[RAG]")
    print(f"Enabled               : {config.rag.enabled}")
    print(f"Provider              : {config.rag.provider}")
    print(f"Host                  : {config.rag.host}")
    print(f"Dataset ID            : {config.rag.dataset_id}")
    print(f"Top K                 : {config.rag.top_k}")

    print("\n[LLM]")
    print(f"Provider              : {config.llm.provider}")
    print(f"Model                 : {config.llm.model}")
    print(f"Max Tokens            : {config.llm.max_tokens}")
    print(f"Temperature            : {config.llm.temperature}")
    print(f"Timeout               : {config.llm.timeout}")
    print(f"Max Retries            : {config.llm.max_retries}")

    print("\n[Cache]")
    print(f"Enabled               : {config.cache.enabled}")

    print("\n[Repair]")
    print(f"Enabled               : {config.repair.enabled}")
    print(f"Model                 : {config.repair.model}")
    print(f"Max Tokens            : {config.repair.max_tokens}")

    print("\n[Checkpoint]")
    print(f"Enabled               : {config.checkpoint.enabled}")
    print(f"Save Every            : {config.checkpoint.save_every}")

    print("\n[Output]")
    print(f"Base Directory        : {config.output.base_dir}")
    print(f"Run ID                : {config.output.run_id}")

    print("\n[Logging]")
    print(f"Level                 : {config.logging.level}")

    print("=" * 60)