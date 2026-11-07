"""
orqa/llm/anthropic_provider.py

Anthropic Claude API Provider。

- API Key 由環境變數 ANTHROPIC_API_KEY 提供。
- 重試：SDK 內建對 408/409/429/5xx 與連線錯誤做 exponential backoff（max_retries 可設定）。
- 不傳送 temperature/top_p/top_k：anthropic SDK 2.x 使用 API 版本 2026-10-21，
  該版本對任何 sampling 參數一律回 400。
- System Prompt 啟用 prompt caching（同一實驗所有題目共用相同 system prompt）。
"""

from __future__ import annotations

import time
from typing import Any, Optional

import anthropic

from orqa.config import AnthropicSection, get_secret
from orqa.llm.base import BaseLLMProvider, FatalLLMError, LLMError, LLMResponse
from orqa.prompts import ChatPrompt


class AnthropicProvider(BaseLLMProvider):

    provider_name = "anthropic"

    def __init__(self, settings: AnthropicSection, *, client: Any = None):
        super().__init__()
        self.settings = settings
        self.client = client or anthropic.Anthropic(
            api_key=get_secret("ANTHROPIC_API_KEY"),
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

    def _system_blocks(self, system: str) -> list[dict]:
        block: dict[str, Any] = {"type": "text", "text": system}
        if self.settings.prompt_caching:
            cache: dict[str, Any] = {"type": "ephemeral"}
            if self.settings.cache_ttl:
                cache["ttl"] = self.settings.cache_ttl
            block["cache_control"] = cache
        return [block]

    def _generate(self, prompt: ChatPrompt, *, json_schema: Optional[dict] = None) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.settings.model_id,
            "max_tokens": self.settings.max_tokens,
            "system": self._system_blocks(prompt.system),
            "messages": [{"role": "user", "content": prompt.user}],
            **self.settings.extra_params,
        }
        if json_schema is not None:
            output_config = dict(params.get("output_config") or {})
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
            params["output_config"] = output_config

        start = time.perf_counter()
        try:
            response = self.client.messages.create(**params)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError) as exc:
            raise FatalLLMError(f"Anthropic API rejected the request: {exc}") from exc
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as exc:
            # SDK 已經依 max_retries 重試過，仍失敗才會到這裡
            raise LLMError(f"Anthropic API call failed after retries: {exc}") from exc
        latency = time.perf_counter() - start

        text = "".join(
            getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text"
        ).strip()
        usage = response.usage
        extra: dict[str, Any] = {"request_id": getattr(response, "_request_id", None)}
        stop_details = getattr(response, "stop_details", None)
        if stop_details is not None:
            extra["stop_details"] = {
                "category": getattr(stop_details, "category", None),
                "explanation": getattr(stop_details, "explanation", None),
            }

        return LLMResponse(
            text=text,
            model_id=getattr(response, "model", None) or self.settings.model_id,
            provider=self.provider_name,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            stop_reason=getattr(response, "stop_reason", None),
            latency_seconds=latency,
            extra=extra,
        )

    def get_model_info(self) -> dict[str, Any]:
        s = self.settings
        return {
            "provider": self.provider_name,
            "model_id": s.model_id,
            "quantization": None,
            "generation_parameters": {
                "max_tokens": s.max_tokens,
                "sampling": "API defaults (sampling parameters are not sent on API version 2026-10-21)",
                "extra_params": dict(s.extra_params),
            },
            "prompt_caching": s.prompt_caching,
            "cache_ttl": s.cache_ttl,
            "timeout": s.timeout,
            "max_retries": s.max_retries,
            "sdk_version": anthropic.__version__,
        }

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
