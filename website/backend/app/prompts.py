"""
System Prompt 組裝。

Baseline = baseline.md + output_format.md
Skill    = 該 Skill 的 wrapper + output_format.md + Skill 全文（不修改內容）+ 結尾提醒

config.yaml 的 prompts.skills 可列出多份 Skill，使用者在前端下拉選單選擇；
啟動時全部載入一次，建立 Session 時把選到的 Skill prompt 快照進 Session。
Skill 來源可為 .skill / .zip / 資料夾（透過既有的 orqa.skill.load_skill 載入），
或單一 .md 檔。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.settings import Settings, SkillSettings

SKILL_HEADER = "\n\n===== OR SKILL (reference manual) =====\n\n"
SKILL_FOOTER = (
    "\n\n===== END OF OR SKILL =====\n\n"
    "Reminder: answer with the structured solution write-up described at the top of these "
    "instructions — never with only an option letter or only a JSON answer record."
)


@dataclass(frozen=True)
class SkillInfo:
    name: str
    source: str
    files: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class SkillPrompt:
    id: str
    label: str
    description: str
    system: str
    info: SkillInfo
    overview: str

    def public(self) -> dict:
        return {"id": self.id, "label": self.label, "description": self.description,
                "name": self.info.name}


@dataclass(frozen=True)
class PromptBundle:
    baseline: str
    skills: dict[str, SkillPrompt]   # 依 config 順序
    default_skill: str

    def skill(self, skill_id: str | None = None) -> SkillPrompt:
        return self.skills[skill_id or self.default_skill]

    def systems_for(self, skill_id: str | None = None) -> dict[str, str]:
        return {"baseline": self.baseline, "skill": self.skill(skill_id).system}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_skill_text(source: Path) -> tuple[str, SkillInfo]:
    if not source.exists():
        raise FileNotFoundError(
            f"找不到 Skill 檔案：{source}。請確認 config.yaml 的 prompts.skills[].source。"
        )
    if source.is_file() and source.suffix.lower() == ".md":
        text = _read(source)
        return text, SkillInfo(source.stem, source.as_posix(), (source.name,), _sha(text))

    from orqa.skill import load_skill  # 既有的 Skill Loader（ZIP / 資料夾）

    skill = load_skill(source)
    return skill.instruction, SkillInfo(skill.name, source.as_posix(), skill.file_names, skill.sha256)


def _build_skill(settings: Settings, cfg: SkillSettings, output_format: str) -> SkillPrompt:
    skill_text, info = load_skill_text(settings.path(cfg.source))
    system = _read(settings.path(cfg.wrapper)) + "\n\n" + output_format + SKILL_HEADER + skill_text + SKILL_FOOTER
    overview = ""
    if cfg.overview:
        overview_path = settings.path(cfg.overview)
        overview = _read(overview_path) if overview_path.is_file() else ""
    return SkillPrompt(cfg.id, cfg.label, cfg.description, system, info, overview)


def build_prompts(settings: Settings) -> PromptBundle:
    p = settings.prompts
    output_format = _read(settings.path(p.output_format))
    baseline = _read(settings.path(p.baseline)) + "\n\n" + output_format
    skills = {cfg.id: _build_skill(settings, cfg, output_format) for cfg in p.skills}
    return PromptBundle(baseline=baseline, skills=skills, default_skill=p.default_skill)
