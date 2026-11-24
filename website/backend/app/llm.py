"""
Claude 呼叫層（Anthropic Python SDK, AsyncAnthropic + streaming）。

- Baseline 與 Skill 共用同一個 ClaudeClient 與同一組參數，只有 system prompt 與歷史不同。
- 一律使用 streaming：長回答不會碰到 HTTP 逾時，前端也能即時顯示。
- 多輪對話：把完整的 response.content（含 thinking 區塊）append 回歷史，不修改既有訊息
  （Sonnet 5.5 的 preserved thinking 會檢查歷史是否被改動）。
- 不傳 temperature / top_p / top_k：目前 API 版本一律回 400。
- refusal：被安全分類器拒答時 stop_reason = "refusal"；可啟用 server-side fallback。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Protocol, Union

import anthropic

from app.settings import LLMSettings

FALLBACK_BETA = "server-side-fallback-2026-07-01"

BranchStatus = Literal["error", "timeout"]


class BranchError(Exception):
    """單一分支失敗。message 為給使用者看的中文說明。"""

    def __init__(self, message: str, *, status: BranchStatus = "error", retryable: bool = True,
                 detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable
        self.detail = detail


@dataclass
class TurnResult:
    text: str
    thinking: str
    content: list[Any]                 # 原樣保存的 response.content，供下一輪送回 API
    model: str
    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    response_time: float = 0.0
    truncated: bool = False
    fallback_used: bool = False
    request_id: str | None = None


@dataclass
class TextDelta:
    text: str
    kind: Literal["text", "thinking"] = "text"


StreamEvent = Union[TextDelta, TurnResult]


class LLMClient(Protocol):
    model_id: str

    def stream_turn(self, system: str, messages: list[dict]) -> AsyncIterator[StreamEvent]:
        ...


def _usage(usage: Any, name: str) -> int:
    return int(getattr(usage, name, 0) or 0)


def _map_error(exc: Exception) -> BranchError:
    detail = str(exc)
    if isinstance(exc, anthropic.AuthenticationError):
        return BranchError("API Key 無效或未設定，請檢查 .env 的 ANTHROPIC_API_KEY。", retryable=False, detail=detail)
    if isinstance(exc, anthropic.PermissionDeniedError):
        return BranchError("這個 API Key 沒有使用此模型的權限。", retryable=False, detail=detail)
    if isinstance(exc, anthropic.NotFoundError):
        return BranchError("找不到指定的模型，請檢查 config.yaml 的 llm.model_id。", retryable=False, detail=detail)
    if isinstance(exc, anthropic.BadRequestError):
        return BranchError("API 拒絕了這個請求（參數或內容不正確）。", retryable=False, detail=detail)
    if isinstance(exc, anthropic.RateLimitError):
        return BranchError("請求太頻繁或額度不足（rate limit），請稍後再重試。", detail=detail)
    if isinstance(exc, anthropic.APITimeoutError):
        return BranchError("連線 Claude API 逾時。", status="timeout", detail=detail)
    if isinstance(exc, anthropic.APIConnectionError):
        return BranchError("無法連線到 Claude API，請檢查網路。", detail=detail)
    if isinstance(exc, anthropic.APIStatusError):
        return BranchError(f"Claude API 暫時無法服務（HTTP {exc.status_code}），請稍後再重試。", detail=detail)
    return BranchError("呼叫模型時發生未預期的錯誤。", detail=detail)


class ClaudeClient:
    def __init__(self, settings: LLMSettings, api_key: str | None, *, client: Any = None):
        self.settings = settings
        self.model_id = settings.model_id
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    def _params(self, system: str, messages: list[dict]) -> dict[str, Any]:
        s = self.settings
        thinking: dict[str, Any] = {"type": "adaptive"}
        if s.show_thinking_summary:
            thinking["display"] = "summarized"
        params: dict[str, Any] = {
            "model": s.model_id,
            "max_tokens": s.max_tokens,
            # system prompt 固定不變 → 快取；top-level cache_control 再快取對話歷史
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "cache_control": {"type": "ephemeral"},
            "messages": messages,
            "thinking": thinking,
            "output_config": {"effort": s.effort},
        }
        if s.refusal_fallback:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    async def stream_turn(self, system: str, messages: list[dict]) -> AsyncIterator[StreamEvent]:
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        deadline = loop.time() + self.settings.branch_timeout
        try:
            async with self._client.beta.messages.stream(**self._params(system, messages)) as stream:
                iterator = stream.__aiter__()
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        event = await asyncio.wait_for(iterator.__anext__(), remaining)
                    except StopAsyncIteration:
                        break
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield TextDelta(delta.text, "text")
                        elif delta.type == "thinking_delta":
                            yield TextDelta(delta.thinking, "thinking")
                final = await stream.get_final_message()
        except asyncio.TimeoutError:
            raise BranchError(
                f"回答超過 {self.settings.branch_timeout:.0f} 秒仍未完成，已停止。",
                status="timeout",
            ) from None
        except BranchError:
            raise
        except anthropic.AnthropicError as exc:
            raise _map_error(exc) from exc

        elapsed = time.perf_counter() - start
        blocks = list(final.content)
        text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text").strip()
        thinking = "\n\n".join(
            (getattr(b, "thinking", "") or "").strip()
            for b in blocks if getattr(b, "type", None) == "thinking"
        ).strip()

        if final.stop_reason == "refusal":
            details = getattr(final, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise BranchError(
                "模型的安全機制拒絕回答這個問題"
                + (f"（類別：{category}）" if category else "")
                + "。可以調整問題的措辭後再試。",
                retryable=False,
            )

        usage = final.usage
        yield TurnResult(
            text=text,
            thinking=thinking,
            content=blocks,
            model=getattr(final, "model", None) or self.model_id,
            stop_reason=final.stop_reason,
            input_tokens=_usage(usage, "input_tokens"),
            output_tokens=_usage(usage, "output_tokens"),
            cache_read_input_tokens=_usage(usage, "cache_read_input_tokens"),
            cache_creation_input_tokens=_usage(usage, "cache_creation_input_tokens"),
            response_time=round(elapsed, 2),
            truncated=final.stop_reason == "max_tokens",
            fallback_used=any(getattr(b, "type", None) == "fallback" for b in blocks),
            request_id=getattr(final, "_request_id", None),
        )
