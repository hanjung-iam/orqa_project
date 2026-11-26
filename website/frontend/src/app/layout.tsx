import type { Metadata, Viewport } from "next";
import "katex/dist/katex.min.css";
import "./globals.css";
import { AppHeader } from "@/components/AppHeader";

export const metadata: Metadata = {
  title: "OptiSkill — OR Reasoning Demo",
  description: "比較一般提示詞與 OR Skill 的 Claude 作業研究推理",
};

// 固定淺色介面：不論系統 / 瀏覽器是否為深色模式，都維持白底
export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#ffffff",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="zh-Hant" className="h-full antialiased">
      <body className="flex min-h-full flex-col">
        <AppHeader />
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
