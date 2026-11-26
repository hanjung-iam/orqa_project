"use client";

import { useEffect } from "react";

/*
 * 「回到首頁初始畫面」的訊號。
 *
 * Next.js（cacheComponents）切換頁面時會用 <Activity> 隱藏並保留頁面狀態，
 * 所以單純連到 "/" 會看到上一次的比較畫面；在首頁點 "/" 則完全沒有動作。
 * Header 點擊 logo / Home 時呼叫 requestHomeReset()：
 *   - 首頁正在顯示 → 監聽器立即執行重設
 *   - 首頁被隱藏（effect 已暫停）→ 留下旗標，首頁再次顯示、effect 重新執行時重設
 */

let requested = false;
const listeners = new Set<() => void>();

export function requestHomeReset() {
  requested = true;
  for (const listener of listeners) listener();
}

export function useHomeReset(onReset: () => void) {
  useEffect(() => {
    const check = () => {
      if (!requested) return;
      requested = false;
      onReset();
    };
    check();
    listeners.add(check);
    return () => {
      listeners.delete(check);
    };
  }, [onReset]);
}
