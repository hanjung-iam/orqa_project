"""
parser.py
負責將 Claude 的原始文字回答轉換成
ORQA evaluation 可以使用的 answer index
"""

VALID_ANSWERS = ("A", "B", "C", "D")


ANSWER_TO_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

INDEX_TO_ANSWER = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

class AnswerParser:

    def __init__(self):
        pass

    def parse(self, raw_response: str) -> int:

        if not isinstance(raw_response, str):
            return -1

        answer = raw_response.strip()
        answer = answer.upper()

        if answer in VALID_ANSWERS:
            return ANSWER_TO_INDEX[answer]

        return -1


def parse_answer(raw_response: str) -> int:
    parser = AnswerParser()
    return parser.parse(raw_response)


def answer_to_index(answer: str) -> int:

    # 將「單一答案字母」轉換成 index。
    if not isinstance(answer, str):
        return -1
    answer = answer.strip().upper()

    return ANSWER_TO_INDEX.get(answer, -1)


def index_to_answer(index: int) -> str | None:

    # 將 index 轉換成「單一答案字母」。
    return INDEX_TO_ANSWER.get(index)