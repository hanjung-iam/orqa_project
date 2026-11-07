"""TransformersProvider.render_chat 的 chat template 相容性（fake tokenizer，不需要 torch）。"""

from __future__ import annotations

import pytest

from orqa.config import TransformersSection
from orqa.llm.transformers_provider import TransformersProvider
from orqa.prompts import ChatPrompt

PROMPT = ChatPrompt(system="SYS RULES", user="QUESTION TEXT", mode="basic")


class FakeTokenizer:
    def __init__(self, allow_system=True, accept_thinking=True, error="System role not supported"):
        self.allow_system, self.accept_thinking, self.error = allow_system, accept_thinking, error
        self.seen = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        if "enable_thinking" in kw and not self.accept_thinking:
            raise TypeError("unexpected keyword enable_thinking")
        if not self.allow_system and any(m["role"] == "system" for m in messages):
            raise ValueError(self.error)
        self.seen.append(messages)
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def provider_with(tok) -> TransformersProvider:
    p = TransformersProvider.__new__(TransformersProvider)  # 不載入模型
    p.settings = TransformersSection()
    p.tokenizer = tok
    return p


def test_system_role_supported():
    p = provider_with(FakeTokenizer())
    assert p.render_chat(PROMPT) == "system:SYS RULES|user:QUESTION TEXT"
    assert not getattr(p, "system_role_merged", False)


def test_system_role_merged_when_template_rejects_it():
    p = provider_with(FakeTokenizer(allow_system=False, accept_thinking=False))
    assert p.render_chat(PROMPT) == "user:SYS RULES\n\nQUESTION TEXT"
    assert p.system_role_merged is True


def test_unrelated_template_errors_are_raised():
    p = provider_with(FakeTokenizer(allow_system=False, error="template is broken"))
    with pytest.raises(ValueError, match="broken"):
        p.render_chat(PROMPT)
