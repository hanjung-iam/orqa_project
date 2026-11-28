"use client";

import { ChevronDown, Loader2, SendHorizontal, Sparkles } from "lucide-react";
import { useState } from "react";
import type { Example, SkillOption } from "@/lib/api";

export function QuestionInput({
  examples,
  maxChars,
  busy,
  disabled,
  skills,
  skillId,
  onSkillChange,
  onSubmit,
}: {
  examples: Example[];
  maxChars: number;
  busy: boolean;
  disabled: boolean;
  skills: SkillOption[];
  skillId: string | null;
  onSkillChange: (id: string) => void;
  onSubmit: (text: string) => void;
}) {
  const selected = skills.find((s) => s.id === skillId);
  const [text, setText] = useState("");
  const tooLong = text.length > maxChars;
  const canSend = !busy && !disabled && text.trim().length > 0 && !tooLong;

  const submit = () => {
    if (canSend) onSubmit(text);
  };

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-neutral-300 bg-white p-3 shadow-sm focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-100 dark:border-neutral-700 dark:bg-neutral-900 dark:focus-within:ring-brand-950">
        {/* 部分瀏覽器擴充功能（表單填寫 / 密碼管理等）會在 React 接手前替輸入框加上屬性
            （例如 data-sharkid），造成 hydration 警告；只忽略這個元素本身的屬性差異。 */}
        <textarea
          suppressHydrationWarning
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          rows={5}
          disabled={busy}
          placeholder="輸入任何作業研究（OR）問題：情境、參數、目標與限制……（Enter 送出，Shift+Enter 換行）"
          className="w-full resize-y bg-transparent px-1 text-[15px] leading-relaxed outline-none placeholder:text-neutral-400"
        />
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <span className={`text-xs tabular-nums ${tooLong ? "font-medium text-red-600" : "text-neutral-400"}`}>
            {text.length.toLocaleString()} / {maxChars.toLocaleString()}
          </span>
          <div className="flex items-center gap-2">
            {skills.length > 0 && (
              <label className="relative inline-flex items-center" title={selected?.description}>
                <span className="sr-only">OptiSkill 端使用的 Skill</span>
                <Sparkles className="pointer-events-none absolute left-2.5 size-4 text-brand-500" />
                <select
                  value={skillId ?? ""}
                  onChange={(e) => onSkillChange(e.target.value)}
                  disabled={busy}
                  className="appearance-none rounded-xl border border-neutral-300 bg-white py-2 pl-8 pr-8 text-sm font-medium text-neutral-700 hover:border-brand-400 focus:border-brand-500 focus:outline-none disabled:opacity-60 dark:border-neutral-600 dark:bg-neutral-900 dark:text-neutral-200"
                >
                  {skills.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute right-2.5 size-4 text-neutral-400" />
              </label>
            )}
            <button
              onClick={submit}
              disabled={!canSend}
              className="inline-flex items-center gap-2 rounded-xl bg-brand-700 px-4 py-2 text-sm font-medium text-white hover:bg-brand-800 disabled:cursor-not-allowed disabled:bg-neutral-300 dark:disabled:bg-neutral-700"
            >
              {busy ? <Loader2 className="size-4 animate-spin" /> : <SendHorizontal className="size-4" />}
              Send
            </button>
          </div>
        </div>
      </div>
      {selected?.description && (
        <p className="mt-2 text-right text-xs text-neutral-500">
          OptiSkill 使用 <span className="font-medium text-neutral-700 dark:text-neutral-300">{selected.label}</span>：{selected.description}
        </p>
      )}

      {examples.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs text-neutral-500">範例題目：</span>
          {examples.map((ex) => (
            <button
              key={ex.id}
              onClick={() => setText(ex.question)}
              title={ex.question}
              disabled={busy}
              className="rounded-full border border-neutral-300 bg-white px-3 py-1 text-sm text-neutral-700 hover:border-brand-400 hover:text-brand-700 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-300"
            >
              {ex.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
