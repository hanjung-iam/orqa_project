"""
skill.py
負責處理使用者提供的本地 Skill

目前支援：
1. ZIP / .skill 檔案
2. 已解壓縮的 directory
"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import zipfile


@dataclass(frozen=True)
class Skill:
    name: str # skill 名稱
    source_path: Path  # 檔案路徑
    source_type: str # zip / directory
    instruction: str
    files: tuple[str, ...]

# 讀取skill並建立skill instruction
class SkillLoader:

    def __init__(
        self,
        skill_path: str | Path,
        source_type: Optional[str] = None,
    ):

        self.skill_path = Path(skill_path)

        if not self.skill_path.exists():
            raise FileNotFoundError(
                f"Skill path does not exist: {self.skill_path}"
            )

        # 判斷source_type
        if source_type is not None:
            source_type = source_type.lower()

        if source_type is None:
            source_type = self._infer_source_type()

        if source_type not in {"zip", "directory"}:
            raise ValueError(
                "Unsupported Skill source_type: "
                f"{source_type}. "
                "Expected 'zip' or 'directory'."
            )

        self.source_type = source_type

    def _infer_source_type(self) -> str:

        if self.skill_path.is_dir():
            return "directory"

        if self.skill_path.is_file():
            if self.skill_path.suffix.lower() in {".zip", ".skill"}:
                return "zip"

        raise ValueError(
            "Cannot determine Skill source type from path: "
            f"{self.skill_path}"
        )

    def load(self) -> Skill:

        if self.source_type == "directory":
            return self._load_from_directory(self.skill_path)

        if self.source_type == "zip":
            return self._load_from_zip(self.skill_path)

        raise RuntimeError(
            f"Unsupported Skill source type: {self.source_type}"
        )

    def _load_from_directory(self, skill_dir: Path) -> Skill:
       
        # Skill 至少包含 SKILL.md
        skill_md = skill_dir / "SKILL.md"

        if not skill_md.exists():
            raise FileNotFoundError(
                f"SKILL.md not found in Skill directory: {skill_dir}"
            )

        # 找出 Skill 中所有 Markdown 文件
        # 使用 rglob() 可以遞迴搜尋子目錄
        markdown_files = sorted(
            path
            for path in skill_dir.rglob("*.md")
            if path.is_file()
        )

        # 將 skill.md 放在第一位
        markdown_files = self._sort_skill_files(
            skill_md,
            markdown_files,
        )

        instruction = self._build_instruction(
            skill_dir,
            markdown_files,
        )

        return Skill(
            name=skill_dir.name,
            source_path=self.skill_path,
            source_type="directory",
            instruction=instruction,
            files=tuple(
                str(path.relative_to(skill_dir))
                for path in markdown_files
            ),
        )


    def _load_from_zip(self, zip_path: Path) -> Skill:

        if not zipfile.is_zipfile(zip_path):
            raise ValueError(
                f"Skill file is not a valid ZIP archive: {zip_path}"
            )


        # 使用TemporaryDirectory
        with TemporaryDirectory(prefix="orqa_skill_") as temp_dir:

            extract_dir = Path(temp_dir)

            # 將 Skill ZIP 解壓縮到暫存資料夾。
            self._extract_zip(zip_path, extract_dir)
            # 找出skill.md
            skill_md = self._find_skill_md(extract_dir)

            skill_root = skill_md.parent

            markdown_files = sorted(
                path
                for path in skill_root.rglob("*.md")
                if path.is_file()
            )

            markdown_files = self._sort_skill_files(
                skill_md,
                markdown_files,
            )

            instruction = self._build_instruction(
                skill_root,
                markdown_files,
            )

            return Skill(
                name=skill_root.name,
                source_path=self.skill_path,
                source_type="zip",
                instruction=instruction,
                files=tuple(
                    str(path.relative_to(skill_root))
                    for path in markdown_files
                ),
            )

    def _extract_zip(self,
        zip_path: Path, extract_dir: Path,) -> None:

        with zipfile.ZipFile(zip_path, "r") as archive:

            for member in archive.infolist():
                # 解壓縮後的路徑
                target_path = (extract_dir / member.filename).resolve()

                try:
                    target_path.relative_to(extract_dir.resolve())
                except ValueError:
                    raise ValueError(
                        f"Unsafe path found in Skill archive: "
                        f"{member.filename}"
                    )

            # 解壓縮所有檔案到 extract_dir
            archive.extractall(extract_dir)

    def _find_skill_md(self, root: Path) -> Path:
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name.lower() == "skill.md"
        ]

        if not candidates:
            raise FileNotFoundError(
                "SKILL.md was not found inside the Skill archive."
            )

        if len(candidates) > 1:
            raise ValueError(
                "Multiple SKILL.md files were found in the "
                "Skill archive:\n"
                + "\n".join(str(path) for path in candidates)
            )

        return candidates[0]

    # 確保 SKILL.md 排在prompt的第一個
    def _sort_skill_files(
        self,
        skill_md: Path,
        markdown_files: list[Path],
    ) -> list[Path]:

        other_files = [
            path
            for path in markdown_files
            if path != skill_md
        ]

        other_files.sort(
            key=lambda path: str(path).lower()
        )

        return [skill_md] + other_files

    def _build_instruction(
        self,
        skill_root: Path,
        markdown_files: list[Path],
    ) -> str:

        sections = []
        system_prompt = (
            "You are an Operations Research Question Answering expert.\n"
            "The following documents constitute your reasoning manual.\n"
            "Treat every section as part of one coherent instruction.\n"
        )
        sections.append(system_prompt)

        # 依照順序逐一加入 Markdown
        # 每個文件前面加入文件名稱
        # 讓模型知道目前讀到哪一份 instruction
        for file_path in markdown_files:

            relative_path = file_path.relative_to(skill_root)

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as file:

                content = file.read().strip()

            sections.append(
                f"\n\n===== {relative_path} =====\n\n"
                f"{content}"
            )

        return "\n".join(sections)

def load_skill(
    skill_path: str | Path,
    source_type: Optional[str] = None,
) -> Skill:

    loader = SkillLoader(
        skill_path=skill_path,
        source_type=source_type,
    )

    return loader.load()