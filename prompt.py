"""
prompts.py
負責建立送給 LLM 的 Prompt。

目前支援四種模式：
    1. Basic
    2. Skill
    3. RAG
    4. Skill + RAG
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from dataset import Question
from skill import Skill


BASIC_INSTRUCTION = (
    "Given the context (following Context:), "
    "select the most appropriate answer to the question "
    "(following Question:). "
    "Answer exactly one letter: 'A', 'B', 'C', or 'D'. "
    "Do not output any other text.\n"
)

# Skill 模式使用的 instruction
# Skill instruction 會在後面接上
SKILL_INSTRUCTION = (
    "Read the Context and Question carefully.\n"
    "Follow the provided ORQA skill.\n"
    "Output exactly one letter: A, B, C, or D.\n"
)

# 放在retrieved chunks 的前面
RAG_CONTEXT_HEADER = (
    "Retrieved Context:\n"
)

# 放在最後引導模型回答
OUTPUT_PREFIX = (
    "\nAnswer: Among A through D, the answer is ("
)

RAG_CHUNK_SEPARATOR = (
    "\n\n--- Retrieved Chunk ---\n\n"
)


@dataclass(frozen=True)
class Prompt:
    instruction: str # system prompt/ instruction
    question: str # user prompt/ question
    output_prefix: str
    mode: str # basic / skill / rag / skill_rag

# 將 Question 物件轉換成 prompt 中的文字
def format_question(question: Question) -> str:

    context = str(question.context)
    question_text = str(question.question)
    options = question.options

    return (
        "Context: "
        + context
        + "\nQuestion: "
        + question_text
        + "\nA. "
        + str(options[0])
        + "\nB. "
        + str(options[1])
        + "\nC. "
        + str(options[2])
        + "\nD. "
        + str(options[3])
    )

# 將 retrieved chunks 放入 prompt 中
def format_rag_context(
    retrieved_chunks: Sequence[str],
) -> str:

    if not retrieved_chunks:
        return ""
    
    # 將所有chunks轉換成string
    chunks = [
        str(chunk).strip()
        for chunk in retrieved_chunks
        if str(chunk).strip()
    ]

    if not chunks:
        return ""

    return (
        RAG_CONTEXT_HEADER
        + RAG_CHUNK_SEPARATOR.join(chunks)
    )

# 建立 prompt中的instruction，根據是否使用 Skill 或 RAG 來決定 instruction 的內容
def build_instruction(
    skill: Optional[Skill] = None,
    retrieved_chunks: Optional[Sequence[str]] = None,
) -> tuple[str, str]:

    use_skill = skill is not None
    use_rag = retrieved_chunks is not None

    # basic
    if not use_skill and not use_rag:

        return (
            BASIC_INSTRUCTION,
            "basic",
        )

    # skill
    if use_skill and not use_rag:

        instruction = (
            SKILL_INSTRUCTION
            + "\n"
            + skill.instruction
        )

        return (
            instruction,
            "skill",
        )

    # RAG
    if not use_skill and use_rag:

        rag_context = format_rag_context(
            retrieved_chunks
        )

        instruction = BASIC_INSTRUCTION

        if rag_context:
            instruction += (
                "\n"
                + rag_context
                + "\n"
            )

        return (
            instruction,
            "rag",
        )

    # skill + rag

    rag_context = format_rag_context(
        retrieved_chunks
    )

    instruction = (
        SKILL_INSTRUCTION
        + "\n"
        + skill.instruction
    )

    if rag_context:
        instruction += (
            "\n\n"
            + rag_context
            + "\n"
        )

    return (
        instruction,
        "skill_rag",
    )


def build_prompt(
    question: Question,
    skill: Optional[Skill] = None,
    retrieved_chunks: Optional[Sequence[str]] = None,
) -> Prompt:

    instruction, mode = build_instruction(
        skill=skill,
        retrieved_chunks=retrieved_chunks,
    )

    question_text = format_question(
        question
    )

    return Prompt(
        instruction=instruction,
        question=question_text,
        output_prefix=OUTPUT_PREFIX,
        mode=mode,
    )

# 將instruction, question合在一起 可作為檢查prompt使用
def render_prompt(prompt: Prompt) -> str:
    
    return (
        prompt.instruction
        + "\n"
        + prompt.question
        + prompt.output_prefix
    )