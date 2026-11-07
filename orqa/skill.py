"""
orqa/skill.py

Skill Loader：讀取 Skill（ZIP / .skill / 目錄），將所有 Markdown 組成 System Prompt。

- 所有 LLM Provider 共用同一份 Skill 文字（不使用 Claude Native Skill API）。
- 載入順序固定：SKILL.md 第一，其餘 .md 依相對路徑（小寫、POSIX 形式）排序。
- 記錄每個檔案與整體的 SHA256，寫入 experiment.json 以利重現。
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

SKILL_PREAMBLE = (
    "You are an Operations Research Question Answering expert.\n"
    "The following documents constitute your reasoning manual.\n"
    "Treat every section as part of one coherent instruction.\n"
)


@dataclass(frozen=True)
class SkillFile:
    path: str      # 相對於 skill root 的 POSIX 路徑
    sha256: str
    content: str


@dataclass(frozen=True)
class Skill:
    name: str
    source_path: str
    source_type: str          # zip | directory
    files: tuple[SkillFile, ...]
    instruction: str
    sha256: str               # 整份 instruction 的 hash

    @property
    def file_names(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "sha256": self.sha256,
            "files": [{"path": f.path, "sha256": f.sha256} for f in self.files],
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _order(files: dict[str, str], skill_md: str) -> list[SkillFile]:
    others = sorted((p for p in files if p != skill_md), key=lambda p: p.lower())
    return [
        SkillFile(path=p, sha256=_sha256_text(files[p]), content=files[p])
        for p in [skill_md, *others]
    ]


def build_instruction(files: list[SkillFile]) -> str:
    sections = [SKILL_PREAMBLE]
    for f in files:
        sections.append(f"\n\n===== {f.path} =====\n\n{f.content.strip()}")
    return "\n".join(sections)


def _infer_source_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file() and path.suffix.lower() in {".zip", ".skill"}:
        return "zip"
    raise ValueError(f"Cannot determine Skill source type from path: {path}")


def _read_markdown(raw: bytes, name: str) -> str:
    # utf-8-sig：容忍 BOM；換行統一為 \n，讓 hash 與平台無關
    return raw.decode("utf-8-sig").replace("\r\n", "\n")


def _find_skill_md(paths: list[str]) -> str:
    candidates = [p for p in paths if PurePosixPath(p).name.lower() == "skill.md"]
    if not candidates:
        raise FileNotFoundError("SKILL.md was not found in the Skill.")
    if len(candidates) > 1:
        raise ValueError(f"Multiple SKILL.md files found: {candidates}")
    return candidates[0]


def _load_zip(path: Path) -> tuple[str, dict[str, str], str]:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Skill file is not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe path found in Skill archive: {n}")
        skill_md = _find_skill_md(names)
        root = PurePosixPath(skill_md).parent
        files = {}
        for n in names:
            p = PurePosixPath(n)
            if p.suffix.lower() != ".md":
                continue
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue  # 不在 skill root 底下的檔案不納入
            files[rel.as_posix()] = _read_markdown(archive.read(n), n)
    name = root.name or path.stem
    return name, files, PurePosixPath(skill_md).relative_to(root).as_posix()


def _load_directory(path: Path) -> tuple[str, dict[str, str], str]:
    all_md = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".md"]
    rel = [p.relative_to(path).as_posix() for p in all_md]
    skill_md = _find_skill_md(rel)
    root = PurePosixPath(skill_md).parent
    files = {}
    for p, r in zip(all_md, rel):
        try:
            key = PurePosixPath(r).relative_to(root).as_posix()
        except ValueError:
            continue
        files[key] = _read_markdown(p.read_bytes(), r)
    name = (path / root).name
    return name, files, PurePosixPath(skill_md).relative_to(root).as_posix()


def load_skill(skill_path: str | Path, source_type: Optional[str] = None) -> Skill:
    path = Path(skill_path)
    if not path.exists():
        raise FileNotFoundError(f"Skill path does not exist: {path}")
    source_type = (source_type or _infer_source_type(path)).lower()
    if source_type == "zip":
        name, files, skill_md = _load_zip(path)
    elif source_type == "directory":
        name, files, skill_md = _load_directory(path)
    else:
        raise ValueError(f"Unsupported Skill source_type: {source_type}")

    ordered = _order(files, skill_md)
    instruction = build_instruction(ordered)
    return Skill(
        name=name,
        source_path=path.as_posix(),
        source_type=source_type,
        files=tuple(ordered),
        instruction=instruction,
        sha256=_sha256_text(instruction),
    )
