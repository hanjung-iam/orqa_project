import type { Metadata } from "next";

export const metadata: Metadata = { title: "About — OptiSkill" };

const STEPS = [
  { title: "輸入問題", body: "在首頁輸入任何作業研究問題，可以是中文或英文，包含情境、參數、目標與限制。不需要遵循 ORQA 資料集的格式，也不需要提供選項。" },
  { title: "或選擇範例", body: "點選「Example 1–5」會把範例題目填入輸入框，但不會自動送出，可以先修改再送出。" },
  { title: "選擇 Skill", body: "Send 旁的下拉選單可選擇 OptiSkill 端要載入哪一份 Skill：Concept Identification（概念辨識）擅長辨認模型結構（ORQA 選擇題），Practical Implementation（實際求解）擅長把 LP / IP 問題解出最佳值並驗證。選定後整個比較（含追問）都使用同一份 Skill；Baseline 不受影響。" },
  { title: "送出並比較", body: "按 Enter 或 Send 後，同一個問題會同時送給兩邊：左邊 Baseline（一般提示詞），右邊 OptiSkill（載入 OR Skill）。兩邊各自獨立執行，先完成的先顯示。" },
  { title: "閱讀分析", body: "OptiSkill 的回答依階段分段（問題理解、模型元素、目標與限制式、模型類型、求解與驗證、假設、結論），每段可以展開或收合。數學式以 LaTeX 呈現。" },
  { title: "分別追問", body: "每一欄下方有自己的追問框，訊息只會傳給那一邊。若某一邊要求補充資訊，就在那一欄回覆；兩邊的對話紀錄完全獨立，不會看到對方的回答。" },
  { title: "失敗與重試", body: "若其中一邊出錯或逾時，另一邊的結果會保留；按下錯誤訊息中的「重試」只會重跑失敗的那一邊。" },
  { title: "新的比較", body: "按「新的比較」會清除目前的對話，開始全新的工作階段，之前的內容不會影響新的比較。" },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-bold tracking-tight">About OptiSkill</h1>
      <p className="mt-4 leading-relaxed text-neutral-600 dark:text-neutral-400">
        OptiSkill 是一個本機執行的展示網頁，用來比較 Claude 在作業研究（Operations Research）問題上，
        使用一般提示詞與載入專題開發的 OR Skill 時，推理與建模方式的差異。兩邊使用相同的模型與輸出格式規則，
        也收到相同的問題；OptiSkill 端另外載入使用者選擇的 Skill，以及讓 Skill 適用於本網頁的使用說明（例如回答的分段結構）。
      </p>

      <h2 className="mt-10 text-xl font-semibold">如何使用</h2>
      <ol className="mt-4 space-y-4">
        {STEPS.map((s, i) => (
          <li key={s.title} className="flex gap-4">
            <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-brand-700 text-sm font-semibold text-white">
              {i + 1}
            </span>
            <div>
              <p className="font-medium">{s.title}</p>
              <p className="mt-0.5 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">{s.body}</p>
            </div>
          </li>
        ))}
      </ol>

      <h2 className="mt-10 text-xl font-semibold">注意事項</h2>
      <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
        <li>模型產生的程式碼只會以文字顯示，網頁不會執行任何程式碼，也不使用外部最佳化求解器。</li>
        <li>模型的計算結果可能有誤，正式使用前請自行驗證。</li>
        <li>對話只保存在後端記憶體中，後端重新啟動或閒置過久後會清除。</li>
        <li>API Key 只存在後端，不會傳到瀏覽器。</li>
      </ul>
    </div>
  );
}
