"""
Transformers Provider 整合測試 —— 需要 NVIDIA GPU，只在 GPU 機器上執行：

    pip install -r requirements-gpu.txt
    huggingface-cli download Qwen/Qwen3.5-9B
    python -m pytest tests/integration/test_transformers_provider.py -m gpu -v -s

可用環境變數 ORQA_TEST_HF_MODEL 改用較小的模型（例如 Qwen/Qwen3.5-4B）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from orqa.config import TransformersSection  # noqa: E402
from orqa.dataset import load_dataset  # noqa: E402
from orqa.llm.transformers_provider import TransformersProvider  # noqa: E402
from orqa.parser import parse_answer  # noqa: E402
from orqa.prompts import build_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def provider():
    model_id = os.getenv("ORQA_TEST_HF_MODEL", "Qwen/Qwen3.5-9B")
    p = TransformersProvider(TransformersSection(model_id=model_id, max_new_tokens=32, batch_size=2))
    yield p
    p.close()


def test_load_and_model_info(provider):
    info = provider.get_model_info()
    print(info)
    assert info["quantization"] == "4bit"
    assert info["gpu"]["max_memory_allocated_gb"] < 12


def test_generate_single_and_batch(provider):
    ds = load_dataset(ROOT / "dataset" / "ORQA_test.jsonl").head(2)
    prompts = [build_prompt(q, "basic") for q in ds]
    chat = provider.render_chat(prompts[0])
    assert prompts[0].system in chat and prompts[0].user in chat

    single = provider.generate(prompts[0])
    print(repr(single.text), parse_answer(single.text))
    assert single.input_tokens > 0 and single.output_tokens > 0

    batch = provider.generate_batch(prompts)
    assert len(batch) == 2
    # 4-bit + left padding 可能造成極小的數值差異，因此只印出比較，不強制相等
    print("single vs batch[0]:", repr(single.text), repr(batch[0].text))
    assert all(r.output_tokens > 0 for r in batch)
    assert provider.get_usage()["calls"] == 3
