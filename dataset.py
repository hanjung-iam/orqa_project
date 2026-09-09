
"""
dataset.py

負責  Dataset 的讀取與資料建立
"""

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Question:
    index: int
    question_type: str
    context: str
    question: str
    options: list[str]
    target_answer: int

class ORQADataset:

    def __init__(self, questions: list[Question]):
        self._questions = questions

    def __len__(self) -> int:

        #回傳 Dataset 中的題目數量
        return len(self._questions)

    def __getitem__(self, index: int) -> Question:
        return self._questions[index]

    def __iter__(self) -> Iterator[Question]:

       # 讓 ORQADataset 可以直接使用 for-loop
        return iter(self._questions)

def _parse_question(item: dict, index: int) -> Question:

    question_type = item["QUESTION_TYPE"]
    context = item["CONTEXT"]
    question = item["QUESTION"]
    options = item["OPTIONS"]
    target_answer = item["TARGET_ANSWER"]

    return Question(
        index=index,
        question_type=question_type,
        context=context,
        question=question,
        options=options,
        target_answer=target_answer,
    )


def load_dataset(path: str | Path) -> ORQADataset:

    path = Path(path)
    questions: list[Question] = []
    question_index  = 0

    with path.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()
            if not line: continue

            item = json.loads(line)

            question = _parse_question(
                item=item,
                index=question_index,
            )

            questions.append(question)

    return ORQADataset(questions)