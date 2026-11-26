"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, streamBranch, type BranchError, type BranchName, type BranchStatus, type Turn } from "./api";

export interface BranchView {
  status: BranchStatus;
  turns: Turn[];
  /** 正在串流中的回答（完成後以後端回傳的最終內容取代） */
  streamingText: string;
  streamingThinking: string;
  /** 送出中或失敗待重試的使用者訊息（尚未寫入 turns） */
  pendingMessage: string | null;
  error: BranchError | null;
}

const emptyBranch = (): BranchView => ({
  status: "idle",
  turns: [],
  streamingText: "",
  streamingThinking: "",
  pendingMessage: null,
  error: null,
});

export const BRANCHES: BranchName[] = ["baseline", "skill"];

export function useComparison() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [question, setQuestion] = useState<string | null>(null);
  /** 這次比較 OptiSkill 端使用的 Skill（建立 Session 時決定，追問沿用） */
  const [skillId, setSkillId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [branches, setBranches] = useState<Record<BranchName, BranchView>>({
    baseline: emptyBranch(),
    skill: emptyBranch(),
  });
  const controllers = useRef<Partial<Record<BranchName, AbortController>>>({});
  const sessionRef = useRef<string | null>(null);

  const update = useCallback((branch: BranchName, fn: (b: BranchView) => BranchView) => {
    setBranches((prev) => ({ ...prev, [branch]: fn(prev[branch]) }));
  }, []);

  const run = useCallback(
    async (sid: string, branch: BranchName, message: string | null, displayMessage: string) => {
      controllers.current[branch]?.abort();
      const controller = new AbortController();
      controllers.current[branch] = controller;
      update(branch, (b) => ({
        ...b,
        status: "loading",
        streamingText: "",
        streamingThinking: "",
        pendingMessage: displayMessage,
        error: null,
      }));
      try {
        await streamBranch(
          sid,
          branch,
          message,
          (e) => {
            if (sessionRef.current !== sid) return; // 已切換到新的比較
            if (e.event === "delta") {
              update(branch, (b) => ({ ...b, streamingText: b.streamingText + e.data.text }));
            } else if (e.event === "thinking") {
              update(branch, (b) => ({ ...b, streamingThinking: b.streamingThinking + e.data.text }));
            } else if (e.event === "done") {
              update(branch, (b) => ({
                ...b,
                status: "completed",
                turns: [...b.turns, { role: "user", content: displayMessage }, e.data.turn],
                streamingText: "",
                streamingThinking: "",
                pendingMessage: null,
                error: null,
              }));
            } else if (e.event === "error") {
              update(branch, (b) => ({
                ...b,
                status: e.data.status,
                error: { message: e.data.message, retryable: e.data.retryable },
              }));
            }
          },
          controller.signal,
        );
      } catch (err) {
        if ((err as Error).name === "AbortError" || sessionRef.current !== sid) return;
        const message = err instanceof ApiError ? err.message : "發生未預期的錯誤。";
        update(branch, (b) => ({ ...b, status: "error", error: { message, retryable: true } }));
      }
    },
    [update],
  );

  /** 開始新的比較：建立 Session（指定 OptiSkill 端的 Skill），兩個分支同時送出相同的問題 */
  const start = useCallback(
    async (text: string, skill?: string) => {
      const message = text.trim();
      if (!message || starting) return;
      setStarting(true);
      setStartError(null);
      try {
        const { session_id, skill_id } = await api.createSession(skill);
        sessionRef.current = session_id;
        setSessionId(session_id);
        setSkillId(skill_id);
        setQuestion(message);
        setBranches({ baseline: emptyBranch(), skill: emptyBranch() });
        // 兩個請求同時發出、各自更新，不互相等待
        for (const branch of BRANCHES) void run(session_id, branch, message, message);
      } catch (err) {
        setStartError(err instanceof ApiError ? err.message : "無法建立比較。");
      } finally {
        setStarting(false);
      }
    },
    [run, starting],
  );

  /** 只對單一分支追問（或回覆它的澄清問題） */
  const followUp = useCallback(
    (branch: BranchName, text: string) => {
      const message = text.trim();
      if (!sessionId || !message || branches[branch].status === "loading") return;
      void run(sessionId, branch, message, message);
    },
    [branches, run, sessionId],
  );

  /** 重試失敗的分支（後端會重送該分支待處理的訊息） */
  const retry = useCallback(
    (branch: BranchName) => {
      const b = branches[branch];
      if (!sessionId || !b.pendingMessage || b.status === "loading") return;
      void run(sessionId, branch, null, b.pendingMessage);
    },
    [branches, run, sessionId],
  );

  /** 清除目前的比較，回到初始畫面 */
  const reset = useCallback(() => {
    for (const branch of BRANCHES) controllers.current[branch]?.abort();
    controllers.current = {};
    const old = sessionRef.current;
    sessionRef.current = null;
    setSessionId(null);
    setQuestion(null);
    setSkillId(null);
    setStartError(null);
    setBranches({ baseline: emptyBranch(), skill: emptyBranch() });
    if (old) api.deleteSession(old).catch(() => undefined);
  }, []);

  useEffect(() => {
    const current = controllers.current;
    return () => {
      for (const branch of BRANCHES) current[branch]?.abort();
    };
  }, []);

  const anyLoading = BRANCHES.some((b) => branches[b].status === "loading");

  return { sessionId, question, skillId, branches, starting, startError, anyLoading, start, followUp, retry, reset };
}
