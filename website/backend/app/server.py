"""
FastAPI 應用程式。

並行方式：前端對 baseline 與 skill 各送一個請求（同時發出），每個請求以 SSE
（text/event-stream）即時回傳該分支的輸出。兩個分支各自獨立：先完成的先顯示、
單邊失敗不影響另一邊、可以只對其中一邊追問或重試。

SSE 事件：
  start    {"branch", "message"}
  thinking {"text"}                 思考摘要片段
  delta    {"text"}                 回答片段
  done     {"turn", "branch"}       完成，turn 為最終內容（以此為準）
  error    {"status", "message", "retryable", "branch"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.llm import BranchError, ClaudeClient, LLMClient, TextDelta, TurnResult
from app.prompts import PromptBundle, build_prompts
from app.sessions import Session, SessionStore
from app.settings import Settings, get_api_key, load_env_files, load_settings

logger = logging.getLogger("optiskill")

SESSION_ID = PathParam(..., pattern=r"^[0-9a-f]{32}$", description="Session ID")
BranchName = Literal["baseline", "skill"]


class MessageIn(BaseModel):
    message: str


class SessionIn(BaseModel):
    skill_id: str | None = None   # 省略時使用 config 的 default_skill


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _load_json(path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 必須是 JSON 陣列")
    return data


class AppState:
    def __init__(self, settings: Settings, prompts: PromptBundle, llm: LLMClient | None,
                 api_key_configured: bool):
        self.settings = settings
        self.prompts = prompts
        self.llm = llm
        self.api_key_configured = api_key_configured
        self.sessions = SessionStore(settings.limits.session_ttl_minutes, settings.limits.max_sessions)


async def run_branch(state: AppState, session: Session, branch: str, message: str) -> AsyncIterator[str]:
    b = session.branches[branch]
    messages = b.history + [{"role": "user", "content": message}]
    yield _sse("start", {"branch": branch, "message": message})
    try:
        if state.llm is None:
            raise BranchError("後端沒有設定 ANTHROPIC_API_KEY，無法呼叫模型。", retryable=False)
        result: TurnResult | None = None
        async for event in state.llm.stream_turn(session.systems[branch], messages):
            if isinstance(event, TextDelta):
                yield _sse("thinking" if event.kind == "thinking" else "delta", {"text": event.text})
            else:
                result = event
        if result is None:
            raise BranchError("模型沒有回傳完整結果。")

        # 成功才寫入歷史（只追加；保存完整 content 供下一輪送回）
        b.history = messages + [{"role": "assistant", "content": result.content}]
        b.turns.append({"role": "user", "content": message})
        assistant_turn = {
            "role": "assistant",
            "content": result.text,
            "thinking": result.thinking or None,
            "model": result.model,
            "response_time": result.response_time,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cache_read_input_tokens": result.cache_read_input_tokens,
            "cache_creation_input_tokens": result.cache_creation_input_tokens,
            "stop_reason": result.stop_reason,
            "truncated": result.truncated,
            "fallback_used": result.fallback_used,
        }
        b.turns.append(assistant_turn)
        b.pending = None
        b.error = None
        b.status = "completed"
        yield _sse("done", {"branch": branch, "turn": assistant_turn})
    except BranchError as exc:
        if exc.detail:
            logger.warning("branch %s failed: %s", branch, exc.detail)
        b.status = exc.status
        b.error = {"message": exc.message, "retryable": exc.retryable}
        yield _sse("error", {"branch": branch, "status": exc.status, "message": exc.message,
                             "retryable": exc.retryable})
    except (asyncio.CancelledError, GeneratorExit):
        b.status = "error"
        b.error = {"message": "連線中斷，回答未完成。", "retryable": True}
        raise
    except Exception:  # noqa: BLE001 — 任何意外都只影響這個分支
        logger.exception("unexpected error in branch %s", branch)
        b.status = "error"
        b.error = {"message": "後端發生未預期的錯誤。", "retryable": True}
        yield _sse("error", {"branch": branch, "status": "error", "message": b.error["message"],
                             "retryable": True})
    finally:
        b.busy = False
        session.touch()


def create_app(settings: Settings | None = None, llm: LLMClient | None = None,
               prompts: PromptBundle | None = None) -> FastAPI:
    load_env_files()
    settings = settings or load_settings()
    prompts = prompts or build_prompts(settings)
    api_key = get_api_key()
    if llm is None and api_key:
        llm = ClaudeClient(settings.llm, api_key)
    state = AppState(settings, prompts, llm, api_key_configured=llm is not None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def periodic_cleanup():
            while True:
                await asyncio.sleep(300)
                state.sessions.cleanup()
        task = asyncio.create_task(periodic_cleanup())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="OptiSkill Demo API", version="1.0.0", lifespan=lifespan)
    app.state.optiskill = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    # ---------- 基本資訊 ----------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "api_key_configured": state.api_key_configured,
                "active_sessions": len(state.sessions)}

    @app.get("/api/config")
    def public_config() -> dict[str, Any]:
        s = settings
        return {
            "model": s.llm.model_id,
            "effort": s.llm.effort,
            "max_tokens": s.llm.max_tokens,
            "show_thinking_summary": s.llm.show_thinking_summary,
            "max_input_chars": s.limits.max_input_chars,
            "max_turns_per_branch": s.limits.max_turns_per_branch,
            "session_ttl_minutes": s.limits.session_ttl_minutes,
            "skills": [sp.public() for sp in prompts.skills.values()],
            "default_skill": prompts.default_skill,
            "api_key_configured": state.api_key_configured,
            "notice": "對話只保存在後端記憶體中，後端重新啟動或閒置過久後會清除。",
        }

    @app.get("/api/examples")
    def examples() -> list[dict[str, Any]]:
        items = _load_json(settings.path(settings.data.examples))
        return [{"id": e["id"], "label": e["label"], "question": e["question"]} for e in items]

    def _skill(skill_id: str | None):
        if skill_id is not None and skill_id not in prompts.skills:
            raise HTTPException(422, f"沒有這個 Skill：{skill_id}")
        return prompts.skill(skill_id)

    @app.get("/api/skill")
    def skill_info(skill_id: str | None = Query(None)) -> dict[str, Any]:
        sp = _skill(skill_id)
        return {**sp.public(), "files": list(sp.info.files), "sha256": sp.info.sha256,
                "overview_markdown": sp.overview}

    # ---------- 專案資料 ----------

    def _materials() -> list[dict[str, Any]]:
        return _load_json(settings.path(settings.data.materials))

    @app.get("/api/materials")
    def materials() -> list[dict[str, Any]]:
        out = []
        for m in _materials():
            path = settings.path(m["file"])
            available = path.is_file()
            out.append({
                "id": m["id"],
                "title": m["title"],
                "category": m.get("category", ""),
                "description": m.get("description", ""),
                "filename": path.name,
                "available": available,
                "size_bytes": path.stat().st_size if available else None,
                "url": f"/api/materials/{m['id']}/file" if available else None,
            })
        return out

    @app.get("/api/materials/{material_id}/file")
    def material_file(material_id: str, download: bool = Query(False)) -> FileResponse:
        for m in _materials():
            if m["id"] == material_id:
                path = settings.path(m["file"])
                if not path.is_file():
                    break
                return FileResponse(path, filename=path.name,
                                    content_disposition_type="attachment" if download else "inline")
        raise HTTPException(404, "找不到這份資料")

    # ---------- 工作階段 ----------

    def _session(session_id: str) -> Session:
        session = state.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "工作階段不存在或已過期（後端可能已重新啟動），請開始新的比較。")
        return session

    @app.post("/api/sessions", status_code=201)
    def create_session(body: SessionIn | None = None) -> dict[str, Any]:
        sp = _skill(body.skill_id if body else None)
        session = state.sessions.create(prompts.systems_for(sp.id), skill_id=sp.id)
        return session.public()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str = SESSION_ID) -> dict[str, Any]:
        return _session(session_id).public()

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str = SESSION_ID) -> None:
        _session(session_id)
        if not state.sessions.delete(session_id):
            raise HTTPException(409, "仍有回答在進行中，請稍後再試。")

    def _start(session: Session, branch: str, message: str) -> StreamingResponse:
        b = session.branches[branch]
        # busy 超過分支逾時仍未釋放，代表串流從未開始（例如瀏覽器提早斷線），視為失效
        stale = time.monotonic() - b.busy_since > settings.llm.branch_timeout + 30
        if b.busy and not stale:
            raise HTTPException(409, "這個分支正在回答中，請等待完成。")
        if b.completed_turns >= settings.limits.max_turns_per_branch:
            raise HTTPException(400, f"每個分支最多 {settings.limits.max_turns_per_branch} 輪對話，請開始新的比較。")
        b.busy = True
        b.busy_since = time.monotonic()
        b.status = "loading"
        b.pending = message
        b.error = None
        return StreamingResponse(
            run_branch(state, session, branch, message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/branches/{branch}/messages")
    def post_message(body: MessageIn, branch: BranchName, session_id: str = SESSION_ID) -> StreamingResponse:
        message = body.message.strip()
        if not message:
            raise HTTPException(422, "訊息不能是空白。")
        if len(message) > settings.limits.max_input_chars:
            raise HTTPException(422, f"訊息太長（上限 {settings.limits.max_input_chars} 字）。")
        return _start(_session(session_id), branch, message)

    @app.post("/api/sessions/{session_id}/branches/{branch}/retry")
    def retry(branch: BranchName, session_id: str = SESSION_ID) -> StreamingResponse:
        session = _session(session_id)
        b = session.branches[branch]
        if b.pending is None or b.status not in ("error", "timeout"):
            raise HTTPException(409, "這個分支沒有需要重試的訊息。")
        return _start(session, branch, b.pending)

    return app
