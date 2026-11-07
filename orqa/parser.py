"""
orqa/parser.py

將模型原始輸出轉換為答案 index（0–3）。無法判斷時回傳 None（Invalid）。

解析順序（全部為確定性規則，不呼叫任何模型）：
1. strict     : 整段輸出就是單一字母 A/B/C/D（大小寫皆可）
2. wrapped    : 單一字母外包標點或 Markdown，例如 "(C)"、"C)"、"C."、"**C**"
3. final_line : 最後一行是單獨的（大寫）字母，例如推理後以 "**B**" 收尾
4. statement: 明確的作答語句，例如 "the answer is (C)"、"Answer: C"、"Final answer: **B**"
                —— 所有語句指向同一個字母才採用；互相矛盾視為 Invalid
5. leading    : 輸出以「字母 + ) . :」開頭，例如 "B) Because ..."、"D. The ..."
   （僅接受大寫，避免把英文冠詞 a 誤判）

Qwen 等模型若輸出 <think>...</think>，會先移除思考段落再解析。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

LETTERS = ("A", "B", "C", "D")
ANSWER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_ANSWER = {v: k for k, v in ANSWER_TO_INDEX.items()}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WRAPPED = re.compile(r"""^[\s(\[{*_`"'“‘]*([ABCDabcd])[\s)\]}*_`"'”’.:,;!]*$""")
# 字母一律用 (?-i:...) 限定大寫，避免把 "the answer is a linear program" 的冠詞 a 當成 A
_STATEMENT = re.compile(
    r"""(?:
            \b(?:final\s+)?answer\s*(?:is|:|=|-)\s*
          | \b(?:correct|best|right)\s+(?:answer|option|choice)\s*(?:is|:|=|-)\s*
          | \b(?:the\s+)?answer\s+(?:would|should)\s+be\s*
          | \bI\s+(?:choose|select|pick)\s+
          | 答案(?:是|為|:|：)\s*
        )
        (?:option\s+)?
        [\s(\[*_`"']*(?-i:([ABCD]))(?![a-zA-Z])
    """,
    re.IGNORECASE | re.VERBOSE,
)
_OPTION_IS = re.compile(
    r"\boption\s+[(\[*_`\"']*(?-i:([ABCD]))[)\]*_`\"']*\s+is\s+(?:the\s+)?(?:correct|best|right)\b",
    re.IGNORECASE,
)
_LEADING = re.compile(r"^[\s*_`]*\(?([ABCD])[)\.:](?:\s|$|\*)")


@dataclass(frozen=True)
class ParseResult:
    prediction: Optional[int]      # 0–3 或 None
    letter: Optional[str]
    method: Optional[str]          # strict | wrapped | final_line | statement | leading | None

    @property
    def is_valid(self) -> bool:
        return self.prediction is not None


INVALID = ParseResult(None, None, None)


def strip_thinking(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    # 若只有開頭 <think> 沒有結尾（被截斷），代表沒有最終答案
    lower = text.lower()
    if "<think>" in lower:
        text = text[: lower.index("<think>")]
    return text.strip()


def _ok(letter: str, method: str) -> ParseResult:
    letter = letter.upper()
    return ParseResult(ANSWER_TO_INDEX[letter], letter, method)


def parse_answer(raw_response: Optional[str]) -> ParseResult:
    if not isinstance(raw_response, str):
        return INVALID
    text = strip_thinking(raw_response)
    if not text:
        return INVALID

    if text.upper() in LETTERS:
        return _ok(text, "strict")

    m = _WRAPPED.match(text)
    if m:
        return _ok(m.group(1), "wrapped")

    # 最後一行是單獨的字母（例如推理之後以 "**B**" 收尾）——視為模型最終的選擇，
    # 優先於內文中的作答語句（內文可能包含 "the intended answer is D, but ..." 之類的假設語氣）
    last_line = text.splitlines()[-1]
    m = _WRAPPED.match(last_line)
    if m and m.group(1).isupper():
        return _ok(m.group(1), "final_line")

    found = {m.group(1) for m in _STATEMENT.finditer(text)}
    found |= {m.group(1) for m in _OPTION_IS.finditer(text)}
    if len(found) == 1:
        return _ok(found.pop(), "statement")
    if len(found) > 1:
        return INVALID  # 互相矛盾

    m = _LEADING.match(text)
    if m:
        return _ok(m.group(1), "leading")

    return INVALID


def index_to_answer(index: Optional[int]) -> Optional[str]:
    if index is None:
        return None
    return INDEX_TO_ANSWER.get(index)
