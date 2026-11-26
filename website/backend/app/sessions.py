"""
記憶體內的比較工作階段（不使用資料庫，後端重啟後會清空）。

每個 Session 有兩個完全獨立的分支（baseline / skill）：
- history：送給 API 的訊息（只追加、不修改；只有成功的回合才寫入）
- turns  ：給前端顯示的對話紀錄
- pending：最近一次失敗的使用者訊息（供 retry 使用，不會寫進 history）
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Branch = Literal["baseline", "skill"]
BRANCHES: tuple[Branch, ...] = ("baseline", "skill")
BranchStatusName = Literal["idle", "loading", "completed", "error", "timeout"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class BranchState:
    history: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    status: BranchStatusName = "idle"
    pending: str | None = None
    error: dict[str, Any] | None = None
    busy: bool = False
    busy_since: float = 0.0

    @property
    def completed_turns(self) -> int:
        return sum(1 for t in self.turns if t["role"] == "assistant")

    def public(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "turns": self.turns,
            "pending_message": self.pending,
            "error": self.error,
        }


@dataclass
class Session:
    session_id: str
    systems: dict[str, str]          # 建立時固定的 system prompt 快照（同一 Session 內不會改變）
    skill_id: str = ""               # OptiSkill 端使用的 Skill（建立時選定，追問沿用）
    created_at: str = field(default_factory=_now_iso)
    last_active: float = field(default_factory=time.monotonic)
    branches: dict[str, BranchState] = field(
        default_factory=lambda: {b: BranchState() for b in BRANCHES}
    )

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def public(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": "active",
            "skill_id": self.skill_id,
            "branches": {name: b.public() for name, b in self.branches.items()},
        }


class SessionStore:
    def __init__(self, ttl_minutes: int, max_sessions: int):
        self.ttl_seconds = ttl_minutes * 60
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def cleanup(self) -> int:
        now = time.monotonic()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self.ttl_seconds and not any(b.busy for b in s.branches.values())
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def create(self, systems: dict[str, str], skill_id: str = "") -> Session:
        self.cleanup()
        while len(self._sessions) >= self.max_sessions:
            idle = [s for s in self._sessions.values() if not any(b.busy for b in s.branches.values())]
            if not idle:
                break
            oldest = min(idle, key=lambda s: s.last_active)
            del self._sessions[oldest.session_id]
        session = Session(session_id=secrets.token_hex(16), systems=dict(systems), skill_id=skill_id)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        self.cleanup()
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def delete(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None or any(b.busy for b in session.branches.values()):
            return False
        del self._sessions[session_id]
        return True
