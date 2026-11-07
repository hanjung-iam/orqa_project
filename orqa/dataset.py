"""
orqa/dataset.py

ORQA 資料集載入。

ORQA JSONL 沒有 index 欄位，index 定義為「非空白行」的 0-based 行號
（與舊版 dataset.py 相同）。因此資料集檔案的 SHA256 會記錄在 experiment.json，
resume 時若檔案內容改變會拒絕續跑，避免 index 錯位。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

VALID_QUESTION_TYPES = tuple(f"Q{i}" for i in range(1, 12))


@dataclass(frozen=True)
class Question:
    index: int
    question_type: str
    context: str
    question: str
    options: tuple[str, ...]
    target_answer: int


class ORQADataset:

    def __init__(self, questions: list[Question], *, name: str, path: Path, sha256: str):
        self._questions = list(questions)
        self._by_index = {q.index: q for q in self._questions}
        if len(self._by_index) != len(self._questions):
            raise ValueError("Duplicate question index in dataset")
        self.name = name
        self.path = path
        self.sha256 = sha256

    def __len__(self) -> int:
        return len(self._questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self._questions)

    def __getitem__(self, position: int) -> Question:
        return self._questions[position]

    def get(self, index: int) -> Question:
        """依 question index（不是位置）取得題目。"""
        return self._by_index[index]

    def has(self, index: int) -> bool:
        return index in self._by_index

    @property
    def indices(self) -> list[int]:
        return [q.index for q in self._questions]

    def head(self, n: int | None) -> "ORQADataset":
        """前 n 題（smoke test 用）。n=None 回傳自己。"""
        if n is None or n >= len(self._questions):
            return self
        return ORQADataset(self._questions[:n], name=self.name, path=self.path, sha256=self.sha256)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parse_question(item: dict, index: int, line_no: int) -> Question:
    try:
        question_type = str(item["QUESTION_TYPE"])
        options = item["OPTIONS"]
        target = item["TARGET_ANSWER"]
        context = str(item["CONTEXT"])
        question = str(item["QUESTION"])
    except KeyError as exc:
        raise ValueError(f"line {line_no}: missing field {exc}") from exc

    if not isinstance(options, list) or len(options) != 4:
        raise ValueError(f"line {line_no}: OPTIONS must be a list of 4 items")
    if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target <= 3:
        raise ValueError(f"line {line_no}: TARGET_ANSWER must be an int in [0, 3], got {target!r}")
    if question_type not in VALID_QUESTION_TYPES:
        raise ValueError(f"line {line_no}: unknown QUESTION_TYPE {question_type!r}")

    return Question(
        index=index,
        question_type=question_type,
        context=context,
        question=question,
        options=tuple(str(o) for o in options),
        target_answer=target,
    )


def load_dataset(path: str | Path, name: str | None = None) -> ORQADataset:
    path = Path(path)
    questions: list[Question] = []
    index = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            questions.append(_parse_question(json.loads(line), index, line_no))
            index += 1
    return ORQADataset(questions, name=name or path.stem, path=path, sha256=file_sha256(path))
