import type { Metadata } from "next";
import { MaterialsList } from "@/components/MaterialsList";

export const metadata: Metadata = { title: "Project Materials — OptiSkill" };

export default function MaterialsPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-bold tracking-tight">Project Materials</h1>
      <p className="mt-3 text-slate-600 dark:text-slate-400">專題的研究海報、簡報與報告（檔案存放在本機 website/assets）。</p>
      <MaterialsList />
    </div>
  );
}
