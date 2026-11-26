"use client";

import { Download, ExternalLink, FileText, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { API_BASE, ApiError, api, type Material } from "@/lib/api";

function formatSize(bytes: number | null) {
  if (bytes == null) return "";
  return bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

export function MaterialsList() {
  const [items, setItems] = useState<Material[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .materials()
      .then(setItems)
      .catch((err) => setError(err instanceof ApiError ? err.message : "無法讀取資料清單。"));
  }, []);

  if (error) return <p className="mt-6 text-sm text-red-600">{error}</p>;
  if (!items)
    return (
      <p className="mt-6 inline-flex items-center gap-2 text-sm text-neutral-500">
        <Loader2 className="size-4 animate-spin" />
        載入中…
      </p>
    );

  return (
    <div className="mt-8 grid gap-4 sm:grid-cols-3">
      {items.map((m) => (
        <div
          key={m.id}
          className={`flex flex-col rounded-2xl border bg-white p-5 shadow-sm dark:bg-neutral-900/70 ${
            m.available ? "border-neutral-200 dark:border-neutral-700" : "border-dashed border-neutral-300 opacity-70 dark:border-neutral-700"
          }`}
        >
          <FileText className="size-8 text-brand-500" />
          <p className="mt-3 text-xs font-medium uppercase tracking-wide text-neutral-500">{m.category}</p>
          <h2 className="mt-1 text-lg font-semibold">{m.title}</h2>
          <p className="mt-1 flex-1 text-sm text-neutral-600 dark:text-neutral-400">{m.description}</p>
          {m.available && m.url ? (
            <>
              <p className="mt-3 truncate text-xs text-neutral-400" title={m.filename}>
                {m.filename} · {formatSize(m.size_bytes)}
              </p>
              <div className="mt-3 flex gap-2">
                <a
                  href={API_BASE + m.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-brand-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-brand-800"
                >
                  <ExternalLink className="size-4" />
                  開啟
                </a>
                <a
                  href={`${API_BASE}${m.url}?download=1`}
                  className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-neutral-300 px-3 py-1.5 text-sm font-medium hover:border-brand-400 dark:border-neutral-600"
                >
                  <Download className="size-4" />
                  下載
                </a>
              </div>
            </>
          ) : (
            <p className="mt-4 text-sm text-neutral-500">尚未提供</p>
          )}
        </div>
      ))}
    </div>
  );
}
