"use client";

import { AlertTriangle, Brain, ChevronRight, CornerDownLeft, RotateCcw, Timer } from "lucide-react";
import { useEffect, useState } from "react";
import type { AssistantTurn, BranchName } from "@/lib/api";
import type { BranchView } from "@/lib/useComparison";
import { Markdown } from "./Markdown";
import { StatusBadge } from "./StatusBadge";

const META: Record<BranchName, { title: string; subtitle: string; accent: string; followLabel: string }> = {
  baseline: {
    title: "Baseline",
    subtitle: "一般提示詞，未載入 OR Skill",
    accent: "border-t-slate-400",
    followLabel: "Baseline",
  },
  skill: {
    title: "OptiSkill",
    subtitle: "載入 OR Skill 的逐步分析",
    accent: "border-t-indigo-500",
    followLabel: "OptiSkill",
  },
};

interface Section {
  heading: string | null;
  body: string;
}

/** 以 `## ` 標題切分段落；標題前的文字作為前言 */
function splitSections(markdown: string): Section[] {
  const sections: Section[] = [];
  let current: Section = { heading: null, body: "" };
  let inFence = false;
  for (const line of markdown.split("\n")) {
    if (line.trimStart().startsWith("```")) inFence = !inFence;
    const m = !inFence && /^##\s+(.+?)\s*#*\s*$/.exec(line);
    if (m) {
      if (current.heading !== null || current.body.trim()) sections.push(current);
      current = { heading: m[1].replace(/\*\*/g, ""), body: "" };
    } else {
      current.body += line + "\n";
    }
  }
  if (current.heading !== null || current.body.trim()) sections.push(current);
  return sections;
}

function SectionedAnswer({ markdown }: { markdown: string }) {
  const sections = splitSections(markdown);
  if (sections.every((s) => s.heading === null)) return <Markdown>{markdown}</Markdown>;
  return (
    <div className="space-y-2">
      {sections.map((s, i) =>
        s.heading === null ? (
          <Markdown key={i}>{s.body}</Markdown>
        ) : (
          <details key={i} open className="group rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
            <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-sm font-semibold text-indigo-700 dark:text-indigo-300">
              <ChevronRight className="size-4 shrink-0 transition-transform group-open:rotate-90" />
              <span>{s.heading}</span>
            </summary>
            <div className="border-t border-slate-100 px-4 py-3 dark:border-slate-800">
              <Markdown>{s.body}</Markdown>
            </div>
          </details>
        ),
      )}
    </div>
  );
}

function Answer({ branch, markdown }: { branch: BranchName; markdown: string }) {
  return branch === "skill" ? <SectionedAnswer markdown={markdown} /> : <Markdown>{markdown}</Markdown>;
}

function Thinking({ text, live }: { text: string; live?: boolean }) {
  if (!text.trim()) return null;
  return (
    <details className="group mb-3 rounded-lg bg-slate-50 text-sm dark:bg-slate-800/60">
      <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-slate-500 dark:text-slate-400">
        <ChevronRight className="size-4 transition-transform group-open:rotate-90" />
        <Brain className="size-4" />
        {live ? "思考中…（點擊查看思考摘要）" : "模型思考摘要"}
      </summary>
      <div className="whitespace-pre-wrap px-4 pb-3 text-slate-600 dark:text-slate-300">{text}</div>
    </details>
  );
}

function Metrics({ turn }: { turn: AssistantTurn }) {
  const totalInput = turn.input_tokens + turn.cache_read_input_tokens + turn.cache_creation_input_tokens;
  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-slate-100 pt-2 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
      <span className="inline-flex items-center gap-1">
        <Timer className="size-3.5" />
        {turn.response_time.toFixed(1)} 秒
      </span>
      <span>
        輸入 {totalInput.toLocaleString()} tokens
        {turn.cache_read_input_tokens > 0 && `（快取 ${turn.cache_read_input_tokens.toLocaleString()}）`}
      </span>
      <span>輸出 {turn.output_tokens.toLocaleString()} tokens</span>
      {turn.fallback_used && <span className="text-amber-600">已由備援模型 {turn.model} 回答</span>}
      {turn.truncated && <span className="text-amber-600">已達輸出上限，回答可能不完整</span>}
    </div>
  );
}

function UserBubble({ text, muted }: { text: string; muted?: boolean }) {
  return (
    <div className="my-3 flex justify-end">
      <div
        className={`max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm px-3 py-2 text-sm ${
          muted ? "bg-slate-100 text-slate-500 dark:bg-slate-800" : "bg-indigo-600 text-white"
        }`}
      >
        {text}
      </div>
    </div>
  );
}

function useElapsed(active: boolean) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!active) return;
    const start = Date.now();
    const id = setInterval(() => setSeconds((Date.now() - start) / 1000), 200);
    return () => {
      clearInterval(id);
      setSeconds(0);
    };
  }, [active]);
  return seconds;
}

function FollowUp({ branch, disabled, onSend }: { branch: BranchName; disabled: boolean; onSend: (t: string) => void }) {
  const [text, setText] = useState("");
  const send = () => {
    if (!text.trim() || disabled) return;
    onSend(text);
    setText("");
  };
  return (
    <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-2 focus-within:border-indigo-400 dark:border-slate-700 dark:bg-slate-900">
      <div className="px-1 pb-1 text-xs font-medium text-slate-500">
        追問 → 只傳給 <span className="text-indigo-600 dark:text-indigo-300">{META[branch].followLabel}</span>
      </div>
      <div className="flex items-end gap-2">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
          disabled={disabled}
          placeholder={disabled ? "請等待回答完成" : "補充資訊或提出追問（Enter 送出，Shift+Enter 換行）"}
          className="min-h-[2.5rem] flex-1 resize-y bg-transparent px-1 text-sm outline-none placeholder:text-slate-400 disabled:cursor-not-allowed"
        />
        <button
          onClick={send}
          disabled={disabled || !text.trim()}
          className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
        >
          <CornerDownLeft className="size-4" />
          送出
        </button>
      </div>
    </div>
  );
}

export function ResponsePanel({
  branch,
  view,
  model,
  onFollowUp,
  onRetry,
}: {
  branch: BranchName;
  view: BranchView;
  model: string | undefined;
  onFollowUp: (text: string) => void;
  onRetry: () => void;
}) {
  const meta = META[branch];
  const loading = view.status === "loading";
  const elapsed = useElapsed(loading);
  const failed = view.status === "error" || view.status === "timeout";
  const hasTurns = view.turns.length > 0;

  return (
    <section className={`flex min-w-0 flex-col rounded-2xl border border-t-4 border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-900/70 ${meta.accent}`}>
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-5 py-3 dark:border-slate-800">
        <div>
          <h2 className="text-lg font-semibold">{meta.title}</h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {meta.subtitle}
            {model && <> · {model}</>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {loading && <span className="text-xs tabular-nums text-slate-500">{elapsed.toFixed(0)} 秒</span>}
          <StatusBadge status={view.status} />
        </div>
      </header>

      <div className="flex-1 px-5 py-4">
        {view.turns.map((turn, i) =>
          turn.role === "user" ? (
            i === 0 ? null : <UserBubble key={i} text={turn.content} />
          ) : (
            <div key={i} className="mb-2">
              {turn.thinking && <Thinking text={turn.thinking} />}
              <Answer branch={branch} markdown={turn.content} />
              <Metrics turn={turn} />
            </div>
          ),
        )}

        {(loading || failed) && view.pendingMessage && hasTurns && (
          <UserBubble text={view.pendingMessage} muted={failed} />
        )}

        {loading && (
          <div>
            <Thinking text={view.streamingThinking} live={!view.streamingText} />
            {view.streamingText ? (
              <Answer branch={branch} markdown={view.streamingText} />
            ) : (
              <div className="space-y-2 py-2" aria-label="等待回應">
                <div className="h-3 w-3/4 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
                <div className="h-3 w-5/6 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
                <div className="h-3 w-2/3 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
              </div>
            )}
          </div>
        )}

        {failed && view.error && (
          <div className="mt-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200">
            <div className="flex items-start gap-2">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" />
              <div className="flex-1">
                <p>{view.error.message}</p>
                {!view.error.retryable && <p className="mt-1 text-xs opacity-80">這個錯誤重試可能仍會失敗。</p>}
              </div>
            </div>
            {view.pendingMessage && (
              <button
                onClick={onRetry}
                className="mt-2 inline-flex items-center gap-1 rounded-md border border-red-300 bg-white px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-100 dark:border-red-800 dark:bg-transparent dark:text-red-200"
              >
                <RotateCcw className="size-3.5" />
                重試 {meta.title}
              </button>
            )}
          </div>
        )}

        {view.status === "idle" && !hasTurns && (
          <p className="py-8 text-center text-sm text-slate-400">送出問題後，回答會顯示在這裡。</p>
        )}

        {hasTurns && <FollowUp branch={branch} disabled={loading} onSend={onFollowUp} />}
      </div>
    </section>
  );
}
