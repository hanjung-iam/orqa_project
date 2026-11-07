"""
orqa/llm

create_provider() 依設定延遲 import 對應的 Provider，
避免使用 Claude 時也必須安裝 torch / transformers（反之亦然）。
"""

from __future__ import annotations

from orqa.config import AnthropicSection, LLMSection, PostAnalysisSection
from orqa.llm.base import BaseLLMProvider, FatalLLMError, LLMError, LLMResponse

__all__ = [
    "BaseLLMProvider", "LLMResponse", "LLMError", "FatalLLMError",
    "create_provider", "create_post_analysis_provider",
]


def create_provider(settings: LLMSection) -> BaseLLMProvider:
    if settings.provider == "anthropic":
        from orqa.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(settings.anthropic)
    if settings.provider == "transformers":
        from orqa.llm.transformers_provider import TransformersProvider
        return TransformersProvider(settings.transformers)
    raise ValueError(f"Unknown LLM provider: {settings.provider}")


def create_post_analysis_provider(settings: PostAnalysisSection) -> BaseLLMProvider:
    from orqa.llm.anthropic_provider import AnthropicProvider
    return AnthropicProvider(AnthropicSection(
        model_id=settings.model_id,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        prompt_caching=False,
        cache_ttl=None,
    ))
