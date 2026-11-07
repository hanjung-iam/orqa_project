"""
orqa/llm/transformers_provider.py

Hugging Face Transformers 本機推論 Provider（NVIDIA CUDA + bitsandbytes 4-bit/8-bit）。

torch / transformers 只在建立此 Provider 時才 import，
因此只用 Claude 的實驗不需要安裝 GPU 套件。

在 GPU 機器上的驗證指令（RTX 4070 SUPER 12GB，尚未在開發機執行）：

    # 1) 安裝（建議 Python 3.12，依 CUDA 版本選擇 torch wheel）
    pip install -r requirements-gpu.txt
    # 2) 下載模型（非 gated，HF_TOKEN 可選）
    huggingface-cli download Qwen/Qwen3.5-9B
    # 3) Provider 整合測試（約需 7–8 GB VRAM）
    python -m pytest tests/integration/test_transformers_provider.py -m gpu -v -s
    # 4) 5 題 smoke test
    python main.py --config config/experiment.yaml --provider transformers --method basic --limit 5
"""

from __future__ import annotations

import gc
import time
from typing import Any, Optional, Sequence

from orqa.config import TransformersSection, get_secret
from orqa.llm.base import BaseLLMProvider, FatalLLMError, LLMError, LLMResponse
from orqa.parser import strip_thinking
from orqa.prompts import ChatPrompt


class TransformersProvider(BaseLLMProvider):

    provider_name = "transformers"

    def __init__(self, settings: TransformersSection):
        super().__init__()
        self.settings = settings
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise FatalLLMError(
                "Transformers provider requires torch + transformers (+ bitsandbytes for quantization). "
                "Install with: pip install -r requirements-gpu.txt"
            ) from exc
        self.torch = torch
        self.transformers = transformers

        if settings.device_map is None and settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise FatalLLMError("CUDA is not available on this machine.")
        if settings.quantization != "none" and not torch.cuda.is_available():
            raise FatalLLMError("bitsandbytes quantization requires an NVIDIA GPU with CUDA.")

        token = get_secret("HF_TOKEN", required=False)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            trust_remote_code=settings.trust_remote_code,
            token=token,
        )
        self.tokenizer.padding_side = "left"  # decoder-only batch generation 需要左側 padding
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model(token)
        self.model.eval()
        self._eos_ids = self._collect_eos_ids()

    # ------------------------------------------------------------------ loading

    def _dtype(self, name: str):
        try:
            return getattr(self.torch, name)
        except AttributeError as exc:
            raise FatalLLMError(f"Unknown torch dtype: {name}") from exc

    def _quantization_config(self):
        s = self.settings
        if s.quantization == "none":
            return None
        bnb = self.transformers.BitsAndBytesConfig
        if s.quantization == "8bit":
            return bnb(load_in_8bit=True)
        return bnb(
            load_in_4bit=True,
            bnb_4bit_quant_type=s.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self._dtype(s.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=s.bnb_4bit_use_double_quant,
        )

    def _load_model(self, token: Optional[str]):
        s = self.settings
        kwargs: dict[str, Any] = {
            "revision": s.revision,
            "trust_remote_code": s.trust_remote_code,
            "dtype": self._dtype(s.dtype),
            "token": token,
        }
        quant = self._quantization_config()
        if quant is not None:
            kwargs["quantization_config"] = quant
        if s.device_map is not None:
            kwargs["device_map"] = s.device_map
            if s.max_memory:
                kwargs["max_memory"] = s.max_memory
        if s.attn_implementation:
            kwargs["attn_implementation"] = s.attn_implementation

        auto = {
            "causal_lm": self.transformers.AutoModelForCausalLM,
            "image_text_to_text": getattr(self.transformers, "AutoModelForImageTextToText", None),
        }
        order = [s.auto_model_class] + [k for k in auto if k != s.auto_model_class]
        last_error: Exception | None = None
        for name in order:
            cls = auto.get(name)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(s.model_id, **kwargs)
                self.loaded_model_class = f"{cls.__name__} → {type(model).__name__}"
                break
            except (ValueError, KeyError) as exc:  # 架構不對應這個 Auto class
                last_error = exc
        else:
            raise FatalLLMError(f"Failed to load {s.model_id}: {last_error}") from last_error

        if s.device_map is None and quant is None:
            model = model.to(s.device)
        return model

    def _collect_eos_ids(self) -> set[int]:
        ids = set()
        for source in (self.tokenizer.eos_token_id, getattr(self.model.generation_config, "eos_token_id", None)):
            if isinstance(source, int):
                ids.add(source)
            elif isinstance(source, (list, tuple)):
                ids.update(int(i) for i in source)
        return ids

    # --------------------------------------------------------------- generation

    def render_chat(self, prompt: ChatPrompt) -> str:
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.settings.enable_thinking, **kwargs
            )
        except TypeError:
            # 不支援 enable_thinking 參數的 chat template
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generation_kwargs(self) -> dict[str, Any]:
        s = self.settings
        kwargs: dict[str, Any] = {
            "max_new_tokens": s.max_new_tokens,
            "do_sample": s.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        for name in ("temperature", "top_p", "top_k", "repetition_penalty"):
            value = getattr(s, name)
            if value is not None:
                kwargs[name] = value
        if not s.do_sample:
            # 避免 generation_config 預設的 temperature/top_p 產生 warning 或被誤用
            kwargs.update({"temperature": None, "top_p": None, "top_k": None})
        return kwargs

    def _count_generated(self, ids: list[int]) -> tuple[int, Optional[str]]:
        for i, tok in enumerate(ids):
            if tok in self._eos_ids:
                return i + 1, "eos"
        n = sum(1 for t in ids if t != self.tokenizer.pad_token_id)
        return n, "max_new_tokens" if n >= self.settings.max_new_tokens else None

    def _run_batch(self, prompts: Sequence[ChatPrompt]) -> list[LLMResponse]:
        torch = self.torch
        texts = [self.render_chat(p) for p in prompts]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**enc, **self._generation_kwargs())
        latency = (time.perf_counter() - start) / len(prompts)

        input_len = enc["input_ids"].shape[1]
        input_counts = enc["attention_mask"].sum(dim=1).tolist()
        responses = []
        for row, n_in in zip(output, input_counts):
            gen_ids = row[input_len:].tolist()
            n_out, stop = self._count_generated(gen_ids)
            raw = self.tokenizer.decode(gen_ids[:n_out], skip_special_tokens=True)
            final = strip_thinking(raw)
            thinking = None
            if "</think>" in raw.lower():
                thinking = raw[: raw.lower().rindex("</think>")].replace("<think>", "").strip() or None
            responses.append(LLMResponse(
                text=final,
                model_id=self.settings.model_id,
                provider=self.provider_name,
                input_tokens=int(n_in),
                output_tokens=int(n_out),
                stop_reason=stop,
                latency_seconds=latency,
                thinking=thinking,
                extra={"raw_output": raw} if raw.strip() != final else {},
            ))
        if self.settings.empty_cache_every_batch:
            self._free_memory()
        return responses

    def _run_batch_with_oom_fallback(self, prompts: Sequence[ChatPrompt]) -> list[LLMResponse]:
        try:
            return self._run_batch(prompts)
        except self.torch.cuda.OutOfMemoryError as exc:
            self._free_memory()
            if len(prompts) == 1:
                raise LLMError(f"CUDA out of memory even with batch size 1: {exc}") from exc
            mid = len(prompts) // 2
            return (self._run_batch_with_oom_fallback(prompts[:mid])
                    + self._run_batch_with_oom_fallback(prompts[mid:]))

    def _generate(self, prompt: ChatPrompt, *, json_schema: Optional[dict] = None) -> LLMResponse:
        if json_schema is not None:
            raise NotImplementedError("TransformersProvider does not support json_schema output")
        return self._run_batch_with_oom_fallback([prompt])[0]

    def generate_batch(self, prompts: Sequence[ChatPrompt]) -> list[LLMResponse]:
        try:
            responses = self._run_batch_with_oom_fallback(list(prompts))
        except Exception:
            with self._lock:
                self._usage.errors += 1
            raise
        with self._lock:
            for r in responses:
                self._usage.add(r)
        return responses

    @property
    def preferred_batch_size(self) -> int:
        return self.settings.batch_size

    # ------------------------------------------------------------------- info

    def _free_memory(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def get_model_info(self) -> dict[str, Any]:
        s = self.settings
        info: dict[str, Any] = {
            "provider": self.provider_name,
            "model_id": s.model_id,
            "revision": s.revision,
            "quantization": s.quantization,
            "quantization_config": (
                {"quant_type": s.bnb_4bit_quant_type, "compute_dtype": s.bnb_4bit_compute_dtype,
                 "double_quant": s.bnb_4bit_use_double_quant} if s.quantization == "4bit" else None
            ),
            "dtype": s.dtype,
            "device_map": s.device_map,
            "loaded_model_class": getattr(self, "loaded_model_class", None),
            "enable_thinking": s.enable_thinking,
            "batch_size": s.batch_size,
            "generation_parameters": {
                k: v for k, v in self._generation_kwargs().items() if k != "pad_token_id"
            },
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers.__version__,
        }
        if self.torch.cuda.is_available():
            info["gpu"] = {
                "name": self.torch.cuda.get_device_name(0),
                "max_memory_allocated_gb": round(self.torch.cuda.max_memory_allocated() / 1024**3, 3),
                "memory_reserved_gb": round(self.torch.cuda.memory_reserved() / 1024**3, 3),
            }
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            info["hf_device_map_devices"] = sorted({str(v) for v in device_map.values()})
        return info

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        self._free_memory()
