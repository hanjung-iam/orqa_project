"""
llm.py
負責將 prompts.py 建立好的 Prompt
送到Claude 取得模型的回答
"""

from dataclasses import dataclass
from typing import Optional

import anthropic

from config import ExperimentConfig
from prompt import Prompt

# 用來儲存模型回傳的結果
@dataclass(frozen=True)
class LLMResponse:

    text: str  # 模型的raw response
    raw_content: list

    input_tokens: int
    output_tokens: int

    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    stop_reason: Optional[str] = None
    model: Optional[str] = None


class ClaudeClient:

    def __init__(
        self,
        config: ExperimentConfig,
    ):

        if not config.llm.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when using Claude."
            )

        self.client = anthropic.Anthropic(
            api_key=config.llm.api_key,
        )
        self.config = config

    # prompt sent to client
    def generate(
        self,
        prompt: Prompt,
    ) -> LLMResponse:

        response = self._create_message(prompt)
        text = self._extract_text(response)

        usage = response.usage

        input_tokens = getattr(
            usage,
            "input_tokens",
            0,
        )

        output_tokens = getattr(
            usage,
            "output_tokens",
            0,
        )

        cache_creation_input_tokens = getattr(
            usage,
            "cache_creation_input_tokens",
            0,
        )

        cache_read_input_tokens = getattr(
            usage,
            "cache_read_input_tokens",
            0,
        )
        return LLMResponse(
            text=text,
            raw_content=response.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            stop_reason=getattr(
                response,
                "stop_reason",
                None,
            ),
            model=getattr(
                response,
                "model",
                self.config.llm.model,
            ),
        )

    def _create_message(
        self,
        prompt: Prompt,
    ):

        user_content = [
            {
                "type": "text",
                "text": prompt.question,
            },
            {
                "type": "text",
                "text": prompt.output_prefix,
            },
        ]

        system = [
            {
                "type": "text",
                "text": prompt.instruction,
                #"cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

        return self.client.messages.create(
            model= self.config.llm.model,
            max_tokens= self.config.llm.max_tokens,
            system= system,
            messages= [
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
        )

    def _extract_text(
        self,
        response,
    ) -> str:

        for block in response.content:

            if getattr(block, "type", None) == "text":
                text = getattr(
                    block,
                    "text",
                    "",
                )
                text = text.strip()
                if text:
                    return text

        raise RuntimeError(
            "Claude response did not contain a "
            "non-empty text block."
        )

    def generate_with_tool(
        self,
        prompt: Prompt,
        model: str,
        tool: dict,
        tool_choice: dict,
    ) -> LLMResponse:
        """
        使用指定模型執行 Tool Calling。

        """
        system = [
            {
                "type": "text",
                "text": prompt.instruction,
                "cache_control": {"type": "ephemeral"}
            }
        ]

        user_content = (
            prompt.question
            + prompt.output_prefix
        )

        response = self.client.messages.create(
            model=model,
            max_tokens=self.config.llm.max_tokens,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            tools=[tool],
            tool_choice=tool_choice,
        )
        usage = response.usage

        cache_creation_input_tokens = getattr(
            usage,
            "cache_creation_input_tokens",
            0,
        )

        cache_read_input_tokens = getattr(
            usage,
            "cache_read_input_tokens",
            0,
        )

        text = ""

        for block in response.content:

            if getattr(block, "type", None) == "text":
                block_text = getattr(block, "text", "")
                if block_text:
                    text += block_text

        return LLMResponse(
            text=text,
            raw_content=response.content,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            stop_reason=response.stop_reason,
            model=response.model,
        )
        


def create_llm_client(
    config: ExperimentConfig,
) -> ClaudeClient:

    return ClaudeClient(config)