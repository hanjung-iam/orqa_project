"use client";

import { memo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

/**
 * 只轉換數學式的分隔符號，讓 remark-math 能辨識：
 *   \[ ... \] → $$ ... $$    \( ... \) → $ ... $
 * 程式碼區塊與行內程式碼保持原樣；數學式內容本身不做任何修改。
 */
export function normalizeMath(src: string): string {
  return src
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g)
    .map((part, i) => {
      if (i % 2 === 1) return part; // 程式碼
      return part
        .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
        .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$${m.trim()}$`);
    })
    .join("");
}

const REMARK = [remarkGfm, remarkMath];
const REHYPE: Parameters<typeof ReactMarkdown>[0]["rehypePlugins"] = [
  [rehypeKatex, { throwOnError: false, strict: "ignore" }],
];

function MarkdownImpl({ children }: { children: string }) {
  return (
    <div className="prose-or">
      <ReactMarkdown
        remarkPlugins={REMARK}
        rehypePlugins={REHYPE}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="table-wrap">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {normalizeMath(children)}
      </ReactMarkdown>
    </div>
  );
}

export const Markdown = memo(MarkdownImpl);
