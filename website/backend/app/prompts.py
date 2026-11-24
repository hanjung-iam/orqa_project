"""
System Prompt 組裝。

Baseline = baseline.md + output_format.md
Skill    = skill_wrapper.md + output_format.md + Skill 全文（不修改內容）+ 結尾提醒

Skill 來源可為 .skill / .zip / 資料夾（透過既有的 orqa.skill.load_skill 載入），
或單一 .md 檔（例如 prompts/generalized_or_skill.md），只要改 config.yaml 即可替換。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.settings import Settings

SKILL_HEADER = "\n\n===== OR SKILL (reference manual) =====\n\n"
SKILL_FOOTER = (
    "\n\n===== END OF OR SKILL =====\n\n"
    "Reminder: in this application there are no answer options. Answer with the structured "
    "solution write-up described at the top of these instructions, never with a single letter."
)


@dataclass(frozen=True)
class SkillInfo:
    name: str
    source: str
    files: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class PromptBundle:
    baseline: str
    skill: str
    skill_info: SkillInfo
    skill_overview: str

    def system_for(self, branch: str) -> str:
        return self.baseline if branch == "baseline" else self.skill


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_skill_text(source: Path) -> tuple[str, SkillInfo]:
    if not source.exists():
        raise FileNotFoundError(
            f"找不到 Skill 檔案：{source}。請確認 config.yaml 的 prompts.skill_source。"
        )
    if source.is_file() and source.suffix.lower() == ".md":
        text = _read(source)
        return text, SkillInfo(source.stem, source.as_posix(), (source.name,), _sha(text))

    from orqa.skill import load_skill  # 既有的 Skill Loader（ZIP / 資料夾）

    skill = load_skill(source)
    return skill.instruction, SkillInfo(skill.name, source.as_posix(), skill.file_names, skill.sha256)


def build_prompts(settings: Settings) -> PromptBundle:
    p = settings.prompts
    output_format = _read(settings.path(p.output_format))
    baseline = _read(settings.path(p.baseline)) + "\n\n" + output_format
    skill_text, info = load_skill_text(settings.path(p.skill_source))
    skill = (
        _read(settings.path(p.skill_wrapper))
        + "\n\n"
        + output_format
        + SKILL_HEADER
        + skill_text
        + SKILL_FOOTER
    )
    overview_path = settings.path(p.skill_overview)
    overview = _read(overview_path) if overview_path.is_file() else ""
    return PromptBundle(baseline=baseline, skill=skill, skill_info=info, skill_overview=overview)
