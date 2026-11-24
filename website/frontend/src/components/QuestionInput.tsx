"use client";

import { Loader2, SendHorizontal } from "lucide-react";
import { useState } from "react";
import type { Example } from "@/lib/api";

export function QuestionInput({
  examples,
  maxChars,
  busy,
  disabled,
  onSubmit,
}: {
  examples: Example[];
  maxChars: number;
  busy: boolean;
  disabled: boolean;
  onSubmit: (text: string) => void;
}) {
  const [text, setText] = useState("");
  const tooLong = text.length > maxChars;
  const canSend = !busy && !disabled && text.trim().length > 0 && !tooLong;

  const submit = () => {
    if (canSend) onSubmit(text);
  };

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-slate-300 bg-white p-3 shadow-sm focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-100 dark:border-slate-700 dark:bg-slate-900 dark:focus-within:ring-indigo-950">
        <textarea
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
          className="w-full resize-y bg-transparent px-1 text-[15px] leading-relaxed outline-none placeholder:text-slate-400"
        />
        <div className="flex items-center justify-between gap-3 pt-1">
          <span className={`text-xs tabular-nums ${tooLong ? "font-medium text-red-600" : "text-slate-400"}`}>
            {text.length.toLocaleString()} / {maxChars.toLocaleString()}
          </span>
          <button
            onClick={submit}
            disabled={!canSend}
            className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
          >
            {busy ? <Loader2 className="size-4 animate-spin" /> : <SendHorizontal className="size-4" />}
            Send
          </button>
        </div>
      </div>

      {examples.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs text-slate-500">範例題目：</span>
          {examples.map((ex) => (
            <button
              key={ex.id}
              onClick={() => setText(ex.question)}
              title={ex.question}
              disabled={busy}
              className="rounded-full border border-slate-300 bg-white px-3 py-1 text-sm text-slate-700 hover:border-indigo-400 hover:text-indigo-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
            >
              {ex.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
