"""
orqa/llm/base.py

統一的 LLM Provider 介面。Inference Engine 只依賴這個模組，
不會 import anthropic / transformers / torch。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from orqa.prompts import ChatPrompt


class LLMError(RuntimeError):
    """可重試或單題失敗的錯誤（例如超過重試次數的 rate limit）。"""


class FatalLLMError(LLMError):
    """設定或權限錯誤，繼續執行沒有意義（例如 API Key 錯誤、模型不存在）。"""


@dataclass(frozen=True)
class LLMResponse:
    text: str                          # 最終輸出（不含思考段落）
    model_id: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    stop_reason: Optional[str] = None
    latency_seconds: float = 0.0
    thinking: Optional[str] = None     # 模型的思考段落（若有）
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStats:
    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_seconds: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.cache_creation_input_tokens += r.cache_creation_input_tokens
        self.cache_read_input_tokens += r.cache_read_input_tokens
        self.latency_seconds += r.latency_seconds

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class BaseLLMProvider(ABC):

    provider_name: str = "base"

    def __init__(self) -> None:
        self._usage = UsageStats()
        self._lock = threading.Lock()

    @abstractmethod
    def _generate(self, prompt: ChatPrompt, *, json_schema: Optional[dict] = None) -> LLMResponse:
        ...

    def generate(self, prompt: ChatPrompt, *, json_schema: Optional[dict] = None) -> LLMResponse:
        try:
            response = self._generate(prompt, json_schema=json_schema)
        except Exception:
            with self._lock:
                self._usage.errors += 1
            raise
        with self._lock:
            self._usage.add(response)
        return response

    def generate_batch(self, prompts: Sequence[ChatPrompt]) -> list[LLMResponse]:
        """預設逐筆執行；支援真正 batch 的 Provider 會覆寫。"""
        return [self.generate(p) for p in prompts]

    @property
    def preferred_batch_size(self) -> int:
        return 1

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        ...

    def get_usage(self) -> dict[str, Any]:
        with self._lock:
            return self._usage.as_dict()

    def close(self) -> None:
        """釋放資源（GPU 記憶體、HTTP 連線）。"""
