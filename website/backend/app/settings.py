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
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class SkillSettings(_Strict):
    id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    label: str                    # 下拉選單顯示的名稱
    description: str = ""         # 簡短說明（適合哪類題目）
    source: str                   # .skill / .zip / 含 SKILL.md 的資料夾 / 單一 .md 檔
    wrapper: str                  # 讓這份 Skill 適用於本網頁的包裝說明
    overview: str = ""            # 「Skill 設計」頁面顯示的說明（不公開完整 prompt）


class PromptSettings(_Strict):
    baseline: str
    output_format: str
    default_skill: str
    skills: list[SkillSettings] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_skills(self) -> "PromptSettings":
        ids = [s.id for s in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("prompts.skills 的 id 不可重複")
        if self.default_skill not in ids:
            raise ValueError(f"prompts.default_skill（{self.default_skill}）不在 prompts.skills 中")
        return self


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
