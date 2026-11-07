"""Phase 1 unit tests: config / dataset / skill / parser / prompts / provider interface."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from orqa.config import ConfigError, apply_overrides, load_config, validate_config, Config
from orqa.dataset import load_dataset
from orqa.parser import parse_answer
from orqa.prompts import build_prompt, OUTPUT_PREFIX, PROMPT_VERSION
from orqa.skill import load_skill

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- config

def test_repo_config_loads_and_validates():
    cfg = load_config(ROOT / "config" / "experiment.yaml", validate=False)
    validate_config(cfg, check_paths=False)
    assert cfg.llm.provider in ("anthropic", "transformers")
    assert cfg.rag.retrieval.similarity_threshold == 0.5
    assert cfg.rag.chunking.chunk_size == 2048 and cfg.rag.chunking.stride == 256
    assert cfg.rag.retrieval.rrf_k == 60 and cfg.rag.retrieval.bm25_top_k == 15
    assert cfg.post_analysis.model_id == "claude-haiku-4-5-20251001"
    assert {d.strategy for d in cfg.rag.documents} == {"case_aware", "heading_based"}


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment:\n  methd: basic\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="methd"):
        load_config(p, validate=False)


def test_type_error_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("rag:\n  retrieval:\n    final_top_k: five\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p, validate=False)


def test_anthropic_sampling_params_rejected():
    cfg = Config()
    bad = dataclasses.replace(cfg, llm=dataclasses.replace(
        cfg.llm, anthropic=dataclasses.replace(cfg.llm.anthropic, extra_params={"temperature": 0})))
    with pytest.raises(ConfigError, match="temperature"):
        validate_config(bad, check_paths=False)


def test_overrides(dataset_file):
    cfg = load_config(None, validate=False)
    cfg = apply_overrides(cfg, validate=False, method="rag", provider="transformers",
                          model_id="Qwen/Qwen3.5-4B", limit=5, dataset_path=str(dataset_file))
    assert cfg.experiment.method == "rag" and cfg.experiment.use_rag and not cfg.experiment.use_skill
    assert cfg.llm.transformers.model_id == "Qwen/Qwen3.5-4B"
    assert cfg.llm.anthropic.model_id == "claude-sonnet-5-5"  # 未被改動
    assert cfg.experiment.limit == 5
    validate_config(cfg)


def test_skill_method_requires_skill_path(dataset_file):
    cfg = apply_overrides(load_config(None, validate=False), validate=False,
                          method="skill", dataset_path=str(dataset_file))
    with pytest.raises(ConfigError, match="skill.path"):
        validate_config(cfg)


# ----------------------------------------------------------------- dataset

def test_dataset_index_is_line_order(dataset_file):
    ds = load_dataset(dataset_file)
    assert len(ds) == 6
    assert ds.indices == list(range(6))
    q = ds.get(3)
    assert q.target_answer == 3 and q.options[0] == "opt3a" and q.question_type == "Q10"
    assert len(ds.sha256) == 64
    assert ds.head(2).indices == [0, 1]


def test_real_dataset_loads():
    ds = load_dataset(ROOT / "dataset" / "ORQA_test.jsonl")
    assert len(ds) == 1468
    assert ds.indices == list(range(1468))


def test_dataset_rejects_bad_target(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"QUESTION_TYPE":"Q1","CONTEXT":"c","QUESTION":"q","OPTIONS":["a","b","c","d"],"TARGET_ANSWER":4}\n')
    with pytest.raises(ValueError, match="TARGET_ANSWER"):
        load_dataset(p)


# ------------------------------------------------------------------- skill

def test_skill_zip_order_and_hash(skill_zip):
    s = load_skill(skill_zip)
    assert s.source_type == "zip" and s.name == "demo-skill"
    assert s.file_names == ("SKILL.md", "Alpha.md", "sub/beta.md", "zeta.md")
    assert s.instruction.index("Main instructions") < s.instruction.index("Alpha notes")
    assert "print(" not in s.instruction
    assert s.sha256 == load_skill(skill_zip).sha256  # 可重現
    assert all(len(f.sha256) == 64 for f in s.files)


def test_skill_directory_matches_zip(skill_zip, tmp_path):
    import zipfile
    out = tmp_path / "extracted"
    zipfile.ZipFile(skill_zip).extractall(out)
    d = load_skill(out / "demo-skill")
    z = load_skill(skill_zip)
    assert d.source_type == "directory"
    assert d.instruction == z.instruction and d.sha256 == z.sha256


@pytest.mark.parametrize("name", ["orqa-formulation-mcq-2.skill", "orqa-formulation-mcq-3.skill"])
def test_real_skills_load(name):
    path = ROOT / "skill" / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    s = load_skill(path)
    assert s.file_names[0] == "SKILL.md" and len(s.files) == 4


# ------------------------------------------------------------------ parser

@pytest.mark.parametrize("text,letter,method", [
    ("C", "C", "strict"),
    (" b \n", "B", "strict"),
    ("(D)", "D", "wrapped"),
    ("C)", "C", "wrapped"),
    ("**A**", "A", "wrapped"),
    ("B.", "B", "wrapped"),
    ("The answer is (C) because the constraint ...", "C", "statement"),
    ("After analysis.\n\nFinal answer: **D**", "D", "statement"),
    ("Answer: B", "B", "statement"),
    ("The correct option is A since ...", "A", "statement"),
    ("Option C is correct because it limits capacity.", "C", "statement"),
    ("答案是 B", "B", "statement"),
    ("B) Because the objective ...", "B", "leading"),
    ("<think>maybe A or B</think>\nC", "C", "strict"),
    # 推理後以單獨字母收尾：以最後一行為準，即使內文有假設語氣的作答語句
    ("That leaves B, unless the intended answer is D, but B is the better reading.\n\n**B**", "B", "final_line"),
    ("Long reasoning about option (C).\n\nD", "D", "final_line"),
])
def test_parser_valid(text, letter, method):
    r = parse_answer(text)
    assert (r.letter, r.method) == (letter, method)
    assert r.prediction == "ABCD".index(letter)


@pytest.mark.parametrize("text", [
    "",
    None,
    "The decision activities are the choices the manager makes.",
    "The answer is a linear program with binary variables.",   # 冠詞 a 不可視為 A
    "The answer is A. Actually, the final answer: C",           # 互相矛盾
    "<think>A seems right but",                                 # 思考被截斷
    "A model of this type is an ILP.",                          # 冠詞開頭
])
def test_parser_invalid(text):
    r = parse_answer(text)
    assert r.prediction is None and not r.is_valid


# ----------------------------------------------------------------- prompts

def test_prompt_modes(dataset_file, skill_zip):
    q = load_dataset(dataset_file).get(1)
    skill = load_skill(skill_zip)

    basic = build_prompt(q, "basic")
    assert basic.version == PROMPT_VERSION == "v2"
    assert "Output exactly one letter: A, B, C, or D." in basic.system
    assert "leter" not in basic.system and "resoning" not in basic.system
    assert basic.user.endswith(OUTPUT_PREFIX)
    assert "Context: Context 1\nQuestion: Question 1?\nA. opt1a\nB. opt1b\nC. opt1c\nD. opt1d" in basic.user
    assert "RETRIEVED" not in basic.user

    sk = build_prompt(q, "skill", skill=skill)
    assert skill.instruction in sk.system and "reasoning and formulation guide" in sk.system
    assert sk.user == basic.user                     # skill 只進 system prompt

    rag = build_prompt(q, "rag", retrieved_chunks=["chunk one", "chunk two"])
    assert "chunk one" not in rag.system             # 檢索內容不放 system prompt
    assert rag.user.index("[Retrieved Chunk 1]\nchunk one") < rag.user.index("Context: Context 1")
    assert "judge its relevance" in rag.system and "not the answer" in rag.system

    both = build_prompt(q, "skill_rag", skill=skill, retrieved_chunks=[])
    assert skill.instruction in both.system and "judge its relevance" in both.system
    assert "No knowledge was retrieved" in both.user


def test_prompt_mode_validation(dataset_file):
    q = load_dataset(dataset_file).get(0)
    with pytest.raises(ValueError):
        build_prompt(q, "skill")
    with pytest.raises(ValueError):
        build_prompt(q, "rag")
    with pytest.raises(ValueError):
        build_prompt(q, "basic", retrieved_chunks=["x"])


# ------------------------------------------------------- anthropic provider

class _FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text="  C  ")],
            usage=SimpleNamespace(input_tokens=11, output_tokens=2,
                                  cache_creation_input_tokens=5, cache_read_input_tokens=0),
            model="claude-sonnet-5-5", stop_reason="end_turn", stop_details=None,
        )


def test_anthropic_provider_request_shape(dataset_file):
    from orqa.config import AnthropicSection
    from orqa.llm.anthropic_provider import AnthropicProvider

    fake = SimpleNamespace(messages=_FakeMessages())
    p = AnthropicProvider(AnthropicSection(extra_params={"output_config": {"effort": "low"}}), client=fake)
    prompt = build_prompt(load_dataset(dataset_file).get(0), "basic")

    r = p.generate(prompt)
    call = fake.messages.calls[0]
    assert r.text == "C" and r.input_tokens == 11 and r.provider == "anthropic"
    assert call["system"][0]["text"] == prompt.system
    assert call["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert call["messages"] == [{"role": "user", "content": prompt.user}]
    assert not {"temperature", "top_p", "top_k"} & set(call)
    assert call["output_config"] == {"effort": "low"}

    p.generate(prompt, json_schema={"type": "object"})
    call2 = fake.messages.calls[1]
    assert call2["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}

    usage = p.get_usage()
    assert usage["calls"] == 2 and usage["input_tokens"] == 22
    assert p.get_model_info()["model_id"] == "claude-sonnet-5-5"


def test_providers_do_not_leak_sdks_into_engine_modules():
    """orqa.llm 套件本身不應 import anthropic / torch（延遲載入）。"""
    import subprocess, sys
    code = ("import sys, orqa.llm, orqa.prompts, orqa.parser; "
            "bad=[m for m in ('anthropic','torch','transformers') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
