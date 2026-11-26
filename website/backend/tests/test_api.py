"""後端 API 測試（使用假的 LLM client，不會呼叫 Claude API）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.llm import BranchError, TextDelta, TurnResult  # noqa: E402
from app.prompts import build_prompts  # noqa: E402
from app.server import create_app  # noqa: E402
from app.settings import load_settings  # noqa: E402


class FakeLLM:
    """依 system prompt 判斷分支；記錄每次收到的 messages。"""

    model_id = "fake-model"

    def __init__(self):
        self.calls: list[tuple[str, list[dict]]] = []
        self.fail_next: dict[str, BranchError] = {}

    @staticmethod
    def branch_of(system: str) -> str:
        return "skill" if "OR SKILL" in system else "baseline"

    async def stream_turn(self, system, messages):
        branch = self.branch_of(system)
        self.calls.append((branch, [dict(m) for m in messages]))
        if branch in self.fail_next:
            raise self.fail_next.pop(branch)
        reply = f"{branch} reply #{len(messages) // 2 + 1}"
        yield TextDelta("thinking...", "thinking")
        for word in reply.split(" "):
            yield TextDelta(word + " ")
        yield TurnResult(text=reply, thinking="thinking...", content=[{"type": "text", "text": reply}],
                         model="fake-model", stop_reason="end_turn", input_tokens=10, output_tokens=5,
                         response_time=0.1)


@pytest.fixture()
def fake():
    return FakeLLM()


@pytest.fixture()
def client(fake, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-SECRET-should-never-leak")
    settings = load_settings()
    app = create_app(settings=settings, llm=fake, prompts=build_prompts(settings))
    with TestClient(app) as c:
        yield c


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def send(client, sid, branch, message):
    r = client.post(f"/api/sessions/{sid}/branches/{branch}/messages", json={"message": message})
    assert r.status_code == 200, r.text
    return parse_sse(r.text)


def new_session(client) -> str:
    r = client.post("/api/sessions")
    assert r.status_code == 201
    return r.json()["session_id"]


def test_health_config_examples_skill(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["model"] == "claude-sonnet-5-5"
    examples = client.get("/api/examples").json()
    assert 3 <= len(examples) <= 5 and all(e["question"] for e in examples)
    skill = client.get("/api/skill").json()
    assert skill["name"] and skill["overview_markdown"]


def test_api_key_never_returned(client):
    sid = new_session(client)
    send(client, sid, "baseline", "hi")
    for path in ["/api/health", "/api/config", "/api/examples", "/api/skill", "/api/materials",
                 f"/api/sessions/{sid}"]:
        assert "SECRET" not in client.get(path).text


def test_both_branches_independent_histories(client, fake):
    sid = new_session(client)
    base = send(client, sid, "baseline", "Q1")
    skill = send(client, sid, "skill", "Q1")
    assert [e for e, _ in base][0] == "start" and base[-1][0] == "done"
    assert base[-1][1]["turn"]["content"] == "baseline reply #1"
    assert skill[-1][1]["turn"]["content"] == "skill reply #1"
    assert any(e == "thinking" for e, _ in skill)

    # 只對 skill 追問
    send(client, sid, "skill", "follow-up for skill")
    last_branch, last_messages = fake.calls[-1]
    assert last_branch == "skill"
    assert [m["role"] for m in last_messages] == ["user", "assistant", "user"]
    flat = json.dumps(last_messages, default=str)
    assert "baseline reply" not in flat  # skill 看不到 baseline 的回答

    state = client.get(f"/api/sessions/{sid}").json()["branches"]
    assert len(state["baseline"]["turns"]) == 2
    assert len(state["skill"]["turns"]) == 4
    assert state["baseline"]["status"] == state["skill"]["status"] == "completed"


def test_one_branch_failure_keeps_other_and_retry(client, fake):
    sid = new_session(client)
    fake.fail_next["skill"] = BranchError("rate limited", retryable=True)
    send(client, sid, "baseline", "Q")
    events = send(client, sid, "skill", "Q")
    assert events[-1][0] == "error" and events[-1][1]["retryable"] is True

    state = client.get(f"/api/sessions/{sid}").json()["branches"]
    assert state["baseline"]["status"] == "completed"
    assert state["skill"]["status"] == "error"
    assert state["skill"]["pending_message"] == "Q"
    assert state["skill"]["turns"] == []  # 失敗的回合不寫入歷史

    baseline_calls = sum(1 for b, _ in fake.calls if b == "baseline")
    retried = client.post(f"/api/sessions/{sid}/branches/skill/retry")
    assert parse_sse(retried.text)[-1][0] == "done"
    assert sum(1 for b, _ in fake.calls if b == "baseline") == baseline_calls  # 成功的分支不會重跑
    # 重試送出的歷史只有一則 user 訊息（沒有重複）
    assert [m["role"] for m in fake.calls[-1][1]] == ["user"]

    # 已完成的分支不能 retry
    assert client.post(f"/api/sessions/{sid}/branches/baseline/retry").status_code == 409


def test_timeout_status(client, fake):
    sid = new_session(client)
    fake.fail_next["baseline"] = BranchError("too slow", status="timeout")
    events = send(client, sid, "baseline", "Q")
    assert events[-1][1]["status"] == "timeout"
    assert client.get(f"/api/sessions/{sid}").json()["branches"]["baseline"]["status"] == "timeout"


def test_sessions_are_isolated(client, fake):
    a, b = new_session(client), new_session(client)
    send(client, a, "baseline", "only in A")
    send(client, b, "baseline", "in B")
    assert [m["content"] for m in fake.calls[-1][1]] == ["in B"]
    assert len(client.get(f"/api/sessions/{b}").json()["branches"]["baseline"]["turns"]) == 2


@pytest.mark.parametrize("payload,status", [
    ({"message": "   "}, 422),
    ({"message": "x" * 9000}, 422),
    ({}, 422),
])
def test_input_validation(client, payload, status):
    sid = new_session(client)
    assert client.post(f"/api/sessions/{sid}/branches/baseline/messages", json=payload).status_code == status


def test_invalid_ids(client):
    sid = new_session(client)
    assert client.post(f"/api/sessions/{sid}/branches/other/messages", json={"message": "x"}).status_code == 422
    assert client.get("/api/sessions/not-a-valid-id").status_code == 422
    assert client.get("/api/sessions/" + "0" * 32).status_code == 404


def test_materials(client):
    items = {m["id"]: m for m in client.get("/api/materials").json()}
    assert items["poster"]["available"] and items["slides"]["available"]
    r = client.get(items["poster"]["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline")
    assert client.get(items["poster"]["url"] + "?download=1").headers["content-disposition"].startswith("attachment")
    assert client.get("/api/materials/nope/file").status_code == 404


def test_skill_prompt_contains_skill_and_baseline_does_not():
    settings = load_settings()
    prompts = build_prompts(settings)
    mcq, clp = prompts.skill("orqa-mcq-2").system, prompts.skill("complexlp-v2").system
    assert "ORQA Formulation MCQ" in mcq and "ORQA Formulation MCQ" not in clp
    assert "WHOLE-UNIT / INTEGER DECISION CHECK" in clp and "WHOLE-UNIT" not in mcq
    for text in (mcq, clp):
        assert "$...$" in text  # 共用輸出格式
    assert "$...$" in prompts.baseline
    assert "OR SKILL" not in prompts.baseline and "ORQA Formulation MCQ" not in prompts.baseline
    assert "Override the Skill's output rules" in clp  # complexlp 的包裝說明覆寫「只輸出 JSON」
    assert prompts.skill().id == settings.prompts.default_skill


def test_config_lists_skills(client):
    cfg = client.get("/api/config").json()
    ids = [s["id"] for s in cfg["skills"]]
    assert ids == ["orqa-mcq-2", "complexlp-v2"] and cfg["default_skill"] == "orqa-mcq-2"
    assert all(s["label"] and s["description"] for s in cfg["skills"])
    info = client.get("/api/skill", params={"skill_id": "complexlp-v2"}).json()
    assert info["id"] == "complexlp-v2" and "Practical Implementation" in info["overview_markdown"]
    assert client.get("/api/skill").json()["id"] == "orqa-mcq-2"
    assert client.get("/api/skill", params={"skill_id": "nope"}).status_code == 422


def test_session_uses_selected_skill(client, fake):
    default = client.post("/api/sessions").json()
    assert default["skill_id"] == "orqa-mcq-2"
    r = client.post("/api/sessions", json={"skill_id": "complexlp-v2"})
    assert r.status_code == 201 and r.json()["skill_id"] == "complexlp-v2"
    sid = r.json()["session_id"]
    app_state = client.app.state.optiskill
    systems = app_state.sessions.get(sid).systems
    assert "WHOLE-UNIT" in systems["skill"] and systems["baseline"] == app_state.prompts.baseline
    assert client.get(f"/api/sessions/{sid}").json()["skill_id"] == "complexlp-v2"
    assert client.post("/api/sessions", json={"skill_id": "../etc"}).status_code == 422
