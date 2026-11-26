"use client";

import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError, api, type SkillInfo, type SkillOption } from "@/lib/api";
import { Markdown } from "./Markdown";

export function SkillOverview() {
  const [skills, setSkills] = useState<SkillOption[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [info, setInfo] = useState<SkillInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .config()
      .then((cfg) => {
        setSkills(cfg.skills);
        setSelected((cur) => cur ?? cfg.default_skill);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "無法讀取 Skill 資訊。"));
  }, []);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    api
      .skill(selected)
      .then((data) => {
        if (!cancelled) setInfo(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "無法讀取 Skill 資訊。");
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  if (error) return <p className="mt-6 text-sm text-red-600">{error}</p>;

  return (
    <>
      {skills.length > 1 && (
        <div role="tablist" className="mt-6 inline-flex flex-wrap gap-1 rounded-xl bg-neutral-100 p-1 dark:bg-neutral-800">
          {skills.map((s) => {
            const active = s.id === selected;
            return (
              <button
                key={s.id}
                role="tab"
                aria-selected={active}
                title={s.description}
                onClick={() => setSelected(s.id)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
                  active
                    ? "bg-white text-brand-700 shadow-sm dark:bg-neutral-900 dark:text-brand-300"
                    : "text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-200"
                }`}
              >
                {s.label}
              </button>
            );
          })}
        </div>
      )}

      {!info || info.id !== selected ? (
        <p className="mt-6 flex items-center gap-2 text-sm text-neutral-500">
          <Loader2 className="size-4 animate-spin" />
          載入中…
        </p>
      ) : (
        <>
          <div className="mt-6 flex flex-wrap gap-2 text-xs">
            <span className="rounded-full bg-brand-50 px-2.5 py-1 font-medium text-brand-700 dark:bg-brand-950 dark:text-brand-300">
              {info.name}
            </span>
            {info.files.map((f) => (
              <span key={f} className="rounded-full border border-neutral-200 px-2.5 py-1 text-neutral-600 dark:border-neutral-700 dark:text-neutral-400">
                {f}
              </span>
            ))}
          </div>
          {info.description && <p className="mt-3 text-sm text-neutral-600 dark:text-neutral-400">適用：{info.description}</p>}
          <div className="mt-8">
            {info.overview_markdown ? (
              <Markdown>{info.overview_markdown}</Markdown>
            ) : (
              <p className="text-sm text-neutral-500">這份 Skill 沒有提供說明頁。</p>
            )}
          </div>
          <p className="mt-10 text-xs text-neutral-400">Skill SHA-256：{info.sha256.slice(0, 16)}…</p>
        </>
      )}
    </>
  );
}
