"""
設定載入。

- 非機密設定：website/backend/config.yaml
- API Key：只從環境變數讀取。啟動時依序載入 website/backend/.env、專案根目錄 .env
  （不覆寫已存在的環境變數），因此可以直接沿用實驗框架原本的 .env。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

BACKEND_DIR = Path(__file__).resolve().parents[1]
WEBSITE_DIR = BACKEND_DIR.parent
REPO_ROOT = WEBSITE_DIR.parent

# 讓後端可以 import 既有的 orqa 套件（只用到 orqa.skill）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMSettings(_Strict):
    model_id: str
    max_tokens: int = Field(16000, ge=256, le=128000)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    show_thinking_summary: bool = True
    refusal_fallback: bool = True
    request_timeout: float = Field(300, gt=0)
    branch_timeout: float = Field(420, gt=0)
    max_retries: int = Field(2, ge=0, le=10)


class PromptSettings(_Strict):
    baseline: str
    skill_wrapper: str
    output_format: str
    skill_source: str
    skill_overview: str


class LimitSettings(_Strict):
    max_input_chars: int = Field(8000, ge=100)
    max_turns_per_branch: int = Field(20, ge=1)
    session_ttl_minutes: int = Field(120, ge=1)
    max_sessions: int = Field(200, ge=1)


class DataSettings(_Strict):
    examples: str
    materials: str


class ServerSettings(_Strict):
    cors_origins: list[str] = ["http://localhost:3000"]


class Settings(_Strict):
    llm: LLMSettings
    prompts: PromptSettings
    limits: LimitSettings = LimitSettings()
    data: DataSettings
    server: ServerSettings = ServerSettings()

    def path(self, relative: str) -> Path:
        """把設定檔中的相對路徑轉為絕對路徑（以 website/backend/ 為基準）。"""
        p = Path(relative)
        return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


def load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for env_file in (BACKEND_DIR / ".env", REPO_ROOT / ".env"):
        if env_file.is_file():
            load_dotenv(env_file, override=False)


def get_api_key() -> str | None:
    value = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    return value or None


def load_settings(path: Path | None = None) -> Settings:
    config_path = path or Path(os.getenv("OPTISKILL_CONFIG", BACKEND_DIR / "config.yaml"))
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    settings = Settings.model_validate(raw)
    # 選填：用環境變數覆寫模型（方便臨時切換，不必改設定檔）
    if os.getenv("CLAUDE_MODEL"):
        settings.llm.model_id = os.environ["CLAUDE_MODEL"].strip()
    return settings
