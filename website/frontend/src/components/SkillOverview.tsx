"use client";

import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError, api, type SkillInfo } from "@/lib/api";
import { Markdown } from "./Markdown";

export function SkillOverview() {
  const [info, setInfo] = useState<SkillInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .skill()
      .then(setInfo)
      .catch((err) => setError(err instanceof ApiError ? err.message : "無法讀取 Skill 資訊。"));
  }, []);

  if (error) return <p className="mt-6 text-sm text-red-600">{error}</p>;
  if (!info)
    return (
      <p className="mt-6 inline-flex items-center gap-2 text-sm text-slate-500">
        <Loader2 className="size-4 animate-spin" />
        載入中…
      </p>
    );

  return (
    <>
      <div className="mt-6 flex flex-wrap gap-2 text-xs">
        <span className="rounded-full bg-indigo-50 px-2.5 py-1 font-medium text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
          {info.name}
        </span>
        {info.files.map((f) => (
          <span key={f} className="rounded-full border border-slate-200 px-2.5 py-1 text-slate-600 dark:border-slate-700 dark:text-slate-400">
            {f}
          </span>
        ))}
      </div>
      <div className="mt-8">
        <Markdown>{info.overview_markdown}</Markdown>
      </div>
      <p className="mt-10 text-xs text-slate-400">Skill SHA-256：{info.sha256.slice(0, 16)}…</p>
    </>
  );
}
