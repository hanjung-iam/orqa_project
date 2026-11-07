from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires an NVIDIA GPU + torch/transformers")
    config.addinivalue_line("markers", "api: calls a paid external API (Anthropic / Voyage)")


def make_rows(n: int = 6) -> list[dict]:
    types = ["Q1", "Q2", "Q3", "Q10", "Q2", "Q11", "Q5", "Q7"]
    return [
        {
            "QUESTION_TYPE": types[i % len(types)],
            "CONTEXT": f"Context {i}",
            "QUESTION": f"Question {i}?",
            "OPTIONS": [f"opt{i}a", f"opt{i}b", f"opt{i}c", f"opt{i}d"],
            "TARGET_ANSWER": i % 4,
            "REASONING": "",
        }
        for i in range(n)
    ]


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    path = tmp_path / "mini.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in make_rows():
            f.write(json.dumps(row) + "\n")
        f.write("\n")  # 空白行應被忽略
    return path


@pytest.fixture
def skill_zip(tmp_path: Path) -> Path:
    path = tmp_path / "demo.skill"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("demo-skill/SKILL.md", "# Demo\nMain instructions.")
        z.writestr("demo-skill/zeta.md", "Zeta notes.")
        z.writestr("demo-skill/Alpha.md", "Alpha notes.")
        z.writestr("demo-skill/sub/beta.md", "Beta notes.")
        z.writestr("demo-skill/script.py", "print('ignored')")
    return path
