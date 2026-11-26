"use client";

import { BookOpen, FileText, Home, Info, Menu, Sparkles, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { requestHomeReset } from "@/lib/homeReset";

const NAV = [
  { href: "/", label: "Home", sub: "比較介面", icon: Home },
  { href: "/about", label: "About", sub: "使用說明", icon: Info },
  { href: "/skill", label: "Skill Design", sub: "Skill 設計", icon: BookOpen },
  { href: "/materials", label: "Project Materials", sub: "海報 · 簡報 · 報告", icon: FileText },
];

export function AppHeader() {
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const pathname = usePathname();
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .config()
      .then((c) => {
        setModel(c.model);
        setOnline(true);
      })
      .catch(() => setOnline(false));
  }, []);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <header className="sticky top-0 z-30 border-t-[3px] border-t-brand-500 border-b border-neutral-200 bg-white/85 backdrop-blur dark:border-b-neutral-800 dark:bg-neutral-950/85">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4 sm:px-6">
        <div ref={menuRef} className="relative">
          <button
            onClick={() => setOpen((v) => !v)}
            aria-label="開啟選單"
            aria-expanded={open}
            className="rounded-lg p-2 hover:bg-neutral-100 dark:hover:bg-neutral-800"
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
          {open && (
            <nav className="absolute left-0 top-12 w-64 rounded-xl border border-neutral-200 bg-white p-1.5 shadow-lg dark:border-neutral-700 dark:bg-neutral-900">
              {NAV.map(({ href, label, sub, icon: Icon }) => {
                const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
                return (
                  <Link
                    key={href}
                    href={href}
                    onClick={() => {
                      setOpen(false);
                      if (href === "/") requestHomeReset();
                    }}
                    className={`flex items-center gap-3 rounded-lg px-3 py-2 ${
                      active ? "bg-brand-50 text-brand-700 dark:bg-brand-950 dark:text-brand-300" : "hover:bg-neutral-50 dark:hover:bg-neutral-800"
                    }`}
                  >
                    <Icon className="size-4" />
                    <span>
                      <span className="block text-sm font-medium">{label}</span>
                      <span className="block text-xs text-neutral-500">{sub}</span>
                    </span>
                  </Link>
                );
              })}
            </nav>
          )}
        </div>

        <Link href="/" onClick={requestHomeReset} className="flex items-center gap-2 text-lg font-bold tracking-tight">
          <Sparkles className="size-5 text-coral-500" />
          OptiSkill
        </Link>

        <div className="ml-auto flex items-center gap-2 text-xs">
          <span
            className={`size-2 rounded-full ${online === null ? "bg-neutral-300" : online ? "bg-emerald-500" : "bg-red-500"}`}
            title={online === false ? "後端未連線" : "後端已連線"}
          />
          <span className="rounded-full border border-neutral-200 px-2.5 py-1 text-neutral-600 dark:border-neutral-700 dark:text-neutral-300">
            Model: {model ?? (online === false ? "後端未連線" : "…")}
          </span>
        </div>
      </div>
    </header>
  );
}
