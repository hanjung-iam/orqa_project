"use client";

import { AlertTriangle, Info, Plus } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type Example, type PublicConfig } from "@/lib/api";
import { useHomeReset } from "@/lib/homeReset";
import { BRANCHES, useComparison } from "@/lib/useComparison";
import { QuestionInput } from "./QuestionInput";
import { ResponsePanel } from "./ResponsePanel";

export function ComparisonView() {
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [backendError, setBackendError] = useState<string | null>(null);
  /** 下拉選單目前選的 Skill（null = 尚未載入設定，使用後端預設） */
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null);
  const cmp = useComparison();
  const skills = config?.skills ?? [];
  const skillId = selectedSkill ?? config?.default_skill ?? null;
  const activeSkill = skills.find((s) => s.id === cmp.skillId);
  const { anyLoading, reset } = cmp;

  // 點 logo / Home：回到初始輸入畫面（回答進行中時先確認）
  const goHome = useCallback(() => {
    if (anyLoading && !window.confirm("仍有回答在進行中，確定要離開目前的比較嗎？")) return;
    reset();
    window.scrollTo({ top: 0 });
  }, [anyLoading, reset]);
  useHomeReset(goHome);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.config(), api.examples()])
      .then(([cfg, ex]) => {
        if (cancelled) return;
        setConfig(cfg);
        setExamples(ex);
        setBackendError(null);
      })
      .catch((err) => {
        if (!cancelled) setBackendError(err instanceof ApiError ? err.message : "無法讀取後端設定。");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const backendDown = backendError !== null;

  return (
    <div className="mx-auto w-full max-w-7xl px-4 pb-16 sm:px-6">
      {backendDown && (
        <div className="mt-6 flex items-start gap-3 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-200">
          <AlertTriangle className="mt-0.5 size-5 shrink-0" />
          <div>
            <p className="font-medium">{backendError}</p>
            <p className="mt-1">
              啟動方式：<code className="rounded bg-amber-100 px-1 dark:bg-amber-900">cd website/backend</code> 後執行{" "}
              <code className="rounded bg-amber-100 px-1 dark:bg-amber-900">uvicorn main:app --reload --port 8000</code>，再重新整理此頁。
            </p>
          </div>
        </div>
      )}
      {config && !config.api_key_configured && (
        <div className="mt-6 flex items-start gap-3 rounded-xl border border-red-300 bg-red-50 p-4 text-sm text-red-900 dark:border-red-800 dark:bg-red-950/50 dark:text-red-200">
          <AlertTriangle className="mt-0.5 size-5 shrink-0" />
          <p>後端沒有設定 ANTHROPIC_API_KEY，無法呼叫模型。請在 .env 中設定後重新啟動後端。</p>
        </div>
      )}

      {cmp.question === null ? (
        <section className="mx-auto flex max-w-3xl flex-col items-center pt-14 text-center sm:pt-20">
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Explore <span className="text-brand-600 dark:text-brand-400">OR Reasoning</span> with AI
          </h1>
          <p className="mt-3 max-w-xl text-neutral-600 dark:text-neutral-400">
            輸入一個作業研究問題，同時比較一般提示詞（Baseline）與載入 OR Skill（OptiSkill）的 Claude 回答。
          </p>
          <div className="mt-8 w-full">
            <QuestionInput
              examples={examples}
              maxChars={config?.max_input_chars ?? 8000}
              busy={cmp.starting}
              disabled={backendDown}
              skills={skills}
              skillId={skillId}
              onSkillChange={setSelectedSkill}
              onSubmit={(text) => cmp.start(text, skillId ?? undefined)}
            />
          </div>
          {cmp.startError && <p className="mt-3 text-sm text-red-600">{cmp.startError}</p>}
          {config && (
            <p className="mt-6 inline-flex items-center gap-1.5 text-xs text-neutral-400">
              <Info className="size-3.5" />
              {config.notice}
            </p>
          )}
        </section>
      ) : (
        <section className="mt-6 rounded-2xl border border-neutral-200 bg-white p-4 shadow-sm dark:border-neutral-700 dark:bg-neutral-900/70">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="flex flex-wrap items-center gap-2 text-xs font-medium uppercase tracking-wide text-neutral-500">
                問題（同時傳給兩邊）
                {activeSkill && (
                  <span
                    title={activeSkill.description}
                    className="rounded-full bg-brand-50 px-2 py-0.5 normal-case tracking-normal text-brand-700 dark:bg-brand-950 dark:text-brand-300"
                  >
                    OptiSkill：{activeSkill.label}
                  </span>
                )}
              </p>
              <p className="mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap text-[15px] leading-relaxed">{cmp.question}</p>
            </div>
            <button
              onClick={() => {
                if (!cmp.anyLoading || window.confirm("仍有回答在進行中，確定要開始新的比較嗎？")) cmp.reset();
              }}
              className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-neutral-300 px-3 py-1.5 text-sm font-medium hover:border-brand-400 hover:text-brand-700 dark:border-neutral-600"
            >
              <Plus className="size-4" />
              新的比較
            </button>
          </div>
        </section>
      )}

      {cmp.question !== null && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          {BRANCHES.map((branch) => (
            <ResponsePanel
              key={branch}
              branch={branch}
              view={cmp.branches[branch]}
              model={config?.model}
              skillLabel={activeSkill?.label}
              onFollowUp={(text) => cmp.followUp(branch, text)}
              onRetry={() => cmp.retry(branch)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
