"""
orqa/storage.py

實驗結果檔案的讀寫。

原則
----
- 原始結果（predictions_raw.jsonl、haiku_analysis.jsonl 等 log 型檔案）只能 append，永遠不覆寫。
  每筆 append 後 flush + fsync，降低中斷時遺失資料的機率。
- 若程式在寫入一行到一半時中斷，檔尾會留下不完整的 JSON：讀取時略過並回報，
  下次 append 前先補上換行，讓新紀錄從新的一行開始（不修改既有內容）。
- 衍生檔案（predictions_post_analysis.jsonl、evaluation_*.json ...）以「寫入暫存檔 → os.replace」
  的方式原子性覆寫，可隨時從原始檔重新產生。
- 所有對應一律使用 index，不依賴檔案順序或筆數。
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

RAW_PREDICTIONS = "predictions_raw.jsonl"
INVALID_ANSWERS = "invalid_answers.jsonl"
RETRIEVALS = "retrievals.jsonl"
INFERENCE_ERRORS = "inference_errors.jsonl"
HAIKU_ANALYSIS = "haiku_analysis.jsonl"
CHECKED_INVALID = "checked_invalid.jsonl"
POST_PREDICTIONS = "predictions_post_analysis.jsonl"
EVALUATION_RAW = "evaluation_raw.json"
EVALUATION_POST = "evaluation_post_analysis.json"
WRONG_RAW = "wrong_predictions_raw.json"
WRONG_POST = "wrong_predictions_post_analysis.json"
EXPERIMENT = "experiment.json"


@dataclass(frozen=True)
class JsonlReadResult:
    records: list[dict[str, Any]]
    corrupt_lines: list[int]          # 無法解析的行號（1-based）


def read_jsonl(path: str | Path) -> JsonlReadResult:
    path = Path(path)
    if not path.exists():
        return JsonlReadResult([], [])
    records, corrupt = [], []
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            corrupt.append(no)
    if corrupt:
        warnings.warn(f"{path}: skipped {len(corrupt)} corrupt line(s): {corrupt[:10]}")
    return JsonlReadResult(records, corrupt)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    if "\n" in line:
        raise ValueError("JSONL record must serialize to a single line")
    with path.open("ab") as f:
        if f.tell() > 0:
            with path.open("rb") as r:
                r.seek(-1, os.SEEK_END)
                if r.read(1) != b"\n":
                    f.write(b"\n")  # 前一筆寫到一半就中斷：新紀錄從新的一行開始
        f.write(line.encode("utf-8") + b"\n")
        f.flush()
        os.fsync(f.fileno())


def index_records(records: Iterable[dict[str, Any]], *,
                  keep: Literal["first", "last"] = "first", name: str = "") -> dict[int, dict[str, Any]]:
    """依 index 建立 dict。keep=first：不可變結果（重複時保留最早的）；keep=last：狀態型紀錄。"""
    out: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for rec in records:
        if "index" not in rec or rec["index"] is None:
            continue
        idx = int(rec["index"])
        if idx in out:
            duplicates += 1
            if keep == "first":
                continue
        out[idx] = rec
    if duplicates and keep == "first":
        warnings.warn(f"{name or 'records'}: {duplicates} duplicate index record(s) ignored (kept first)")
    return out


def write_json_atomic(path: str | Path, data: Any) -> None:
    _atomic_write(Path(path), json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    _atomic_write(Path(path), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class ExperimentStorage:
    """單一實驗目錄 results/<experiment_id>/ 的所有檔案。"""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def path(self, name: str) -> Path:
        return self.dir / name

    # ----------------------------------------------------------- raw (append-only)

    def raw_predictions(self) -> dict[int, dict[str, Any]]:
        return index_records(read_jsonl(self.path(RAW_PREDICTIONS)).records, keep="first",
                             name=RAW_PREDICTIONS)

    def completed_indices(self) -> set[int]:
        return set(self.raw_predictions())

    def append_raw_prediction(self, record: dict[str, Any], *, completed: set[int]) -> None:
        idx = int(record["index"])
        if idx in completed:
            raise RuntimeError(f"Refusing to write a second raw prediction for index {idx}")
        append_jsonl(self.path(RAW_PREDICTIONS), record)
        completed.add(idx)

    def append(self, name: str, record: dict[str, Any]) -> None:
        if name in (RAW_PREDICTIONS,):
            raise RuntimeError("use append_raw_prediction() for raw predictions")
        append_jsonl(self.path(name), record)

    def latest(self, name: str) -> dict[int, dict[str, Any]]:
        return index_records(read_jsonl(self.path(name)).records, keep="last", name=name)

    # ---------------------------------------------------- derived (atomic rewrite)

    def write_derived_jsonl(self, name: str, records: Iterable[dict[str, Any]]) -> None:
        if name in (RAW_PREDICTIONS, HAIKU_ANALYSIS, INVALID_ANSWERS, RETRIEVALS, INFERENCE_ERRORS):
            raise RuntimeError(f"{name} is append-only and must not be rewritten")
        write_jsonl_atomic(self.path(name), records)

    def write_json(self, name: str, data: Any) -> None:
        write_json_atomic(self.path(name), data)

    def read_json(self, name: str) -> Any:
        return read_json(self.path(name))

    def exists(self, name: str) -> bool:
        return self.path(name).exists()
