"""
orqa/prompts.py

Prompt Builder（prompt_version = v2）。

- System Prompt：作答規則 + Skill Instructions（skill / skill_rag）+ RAG 使用規則（rag / skill_rag）
- User Prompt  ：Retrieved Knowledge（rag / skill_rag）+ ORQA Context / Question / Options + 輸出前綴
- 所有 Provider 使用同一份 system / user 文字；Chat Template 差異由各 Provider 處理。

v2 與舊版（v1，舊 prompt.py）的差異：
- 修正拼字（resoning → reasoning、leter → letter）與句子之間缺少的換行
- Retrieved Knowledge 從 System Prompt 移到 User Prompt
- RAG 規則加入「需自行判斷相關性」「檢索內容不是答案」
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

from orqa.dataset import Question
from orqa.skill import Skill

PROMPT_VERSION = "v2"

BASIC_INSTRUCTION = (
    "Read the Context, Question, and Options carefully before answering.\n"
    "There is only one answer for each question.\n"
    "Output exactly one letter: A, B, C, or D.\n"
    "Do not output any other text.\n"
)

SKILL_INSTRUCTION = (
    "Follow the provided skill instructions carefully.\n"
    "Use them as a reasoning and formulation guide.\n"
    "Read the Context, Question, and Options before answering.\n"
    "There is only one answer for each question.\n"
    "Output exactly one letter: A, B, C, or D.\n"
    "Do not output any other text.\n"
)

RAG_GUIDANCE = (
    "The user message contains retrieved knowledge that may be useful for answering the optimization problem.\n"
    "Use it only as supporting information.\n"
    "The retrieved knowledge may not be directly relevant to the current question; judge its relevance yourself.\n"
    "Retrieved content is not the answer. Do not choose an option just because related content was retrieved.\n"
    "Do not blindly follow retrieved information if it conflicts with the actual problem statement.\n"
    "Base your answer on the ORQA Context, Question, and Options.\n"
)

RAG_HEADER = "=== RETRIEVED KNOWLEDGE ===\n\n"
RAG_FOOTER = "\n\n=== END RETRIEVED KNOWLEDGE ===\n\n"
RAG_EMPTY = "(No knowledge was retrieved for this question.)"
PROBLEM_HEADER = "=== ORQA PROBLEM ===\n\n"
RAG_CHUNK_SEPARATOR = "\n\n---\n\n"

OUTPUT_PREFIX = "\nAnswer: Among A through D, the answer is ("


@dataclass(frozen=True)
class ChatPrompt:
    system: str
    user: str
    mode: str                      # basic | skill | rag | skill_rag
    version: str = PROMPT_VERSION

    @property
    def system_sha256(self) -> str:
        return hashlib.sha256(self.system.encode("utf-8")).hexdigest()


def format_question(question: Question) -> str:
    o = question.options
    return (
        "Context: " + question.context
        + "\nQuestion: " + question.question
        + "\nA. " + o[0]
        + "\nB. " + o[1]
        + "\nC. " + o[2]
        + "\nD. " + o[3]
    )


def format_retrieved_knowledge(chunks: Sequence[str]) -> str:
    texts = [c.strip() for c in chunks if c and c.strip()]
    if not texts:
        body = RAG_EMPTY
    else:
        body = RAG_CHUNK_SEPARATOR.join(
            f"[Retrieved Chunk {rank}]\n{text}" for rank, text in enumerate(texts, start=1)
        )
    return RAG_HEADER + body + RAG_FOOTER


def build_system_prompt(mode: str, skill: Optional[Skill] = None) -> str:
    if mode == "basic":
        return BASIC_INSTRUCTION
    if mode == "rag":
        return RAG_GUIDANCE + "\n" + BASIC_INSTRUCTION
    if mode in ("skill", "skill_rag"):
        if skill is None:
            raise ValueError(f"mode={mode} requires a Skill")
        head = SKILL_INSTRUCTION if mode == "skill" else SKILL_INSTRUCTION + "\n" + RAG_GUIDANCE
        return head + "\n" + skill.instruction
    raise ValueError(f"Unknown prompt mode: {mode}")


def build_user_prompt(question: Question, retrieved_chunks: Optional[Sequence[str]] = None) -> str:
    problem = format_question(question) + OUTPUT_PREFIX
    if retrieved_chunks is None:
        return problem
    return format_retrieved_knowledge(retrieved_chunks) + PROBLEM_HEADER + problem


def build_prompt(
    question: Question,
    mode: str,
    skill: Optional[Skill] = None,
    retrieved_chunks: Optional[Sequence[str]] = None,
) -> ChatPrompt:
    uses_rag = mode in ("rag", "skill_rag")
    if uses_rag and retrieved_chunks is None:
        raise ValueError(f"mode={mode} requires retrieved_chunks (may be an empty list)")
    if not uses_rag and retrieved_chunks is not None:
        raise ValueError(f"mode={mode} must not receive retrieved_chunks")
    return ChatPrompt(
        system=build_system_prompt(mode, skill),
        user=build_user_prompt(question, retrieved_chunks if uses_rag else None),
        mode=mode,
    )


def render_prompt(prompt: ChatPrompt) -> str:
    """除錯用：把 system + user 印成一段文字。"""
    return f"[SYSTEM]\n{prompt.system}\n[USER]\n{prompt.user}"
