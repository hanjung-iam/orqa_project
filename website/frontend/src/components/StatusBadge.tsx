import { AlertTriangle, CheckCircle2, Circle, Clock, Loader2 } from "lucide-react";
import type { BranchStatus } from "@/lib/api";

const STYLES: Record<BranchStatus, { label: string; className: string; icon: React.ReactNode }> = {
  idle: { label: "Idle", className: "bg-neutral-100 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-300", icon: <Circle className="size-3.5" /> },
  loading: { label: "Loading", className: "bg-brand-50 text-brand-700 dark:bg-brand-950 dark:text-brand-300", icon: <Loader2 className="size-3.5 animate-spin" /> },
  completed: { label: "Completed", className: "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300", icon: <CheckCircle2 className="size-3.5" /> },
  error: { label: "Error", className: "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300", icon: <AlertTriangle className="size-3.5" /> },
  timeout: { label: "Timeout", className: "bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-300", icon: <Clock className="size-3.5" /> },
};

export function StatusBadge({ status }: { status: BranchStatus }) {
  const s = STYLES[status];
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${s.className}`}>
      {s.icon}
      {s.label}
    </span>
  );
}
