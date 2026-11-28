# OptiSkill Interactive Demo

在本機執行的互動式網頁：輸入作業研究（OR）問題，同時比較兩種 Claude 回答方式。

| 欄位 | 說明 |
|---|---|
| **Baseline** | 一般提示詞，不載入 OR Skill |
| **OptiSkill** | 載入專題開發的 OR Skill（`skill/orqa-formulation-mcq-2.skill`） |

兩邊使用相同的模型（`claude-sonnet-5-5`）、相同的參數與輸出格式規則，收到相同的問題；兩邊各自獨立執行、各自保存對話紀錄。

---

## 1. 資料夾結構

```text
website/
├── assets/                 # 海報、簡報、報告（PDF）
├── backend/                # FastAPI（Python）
│   ├── main.py             # 入口：uvicorn main:app
│   ├── config.yaml         # 模型、參數、路徑、限制（不含 API Key）
│   ├── app/
│   │   ├── settings.py     # 設定載入；API Key 只從環境變數讀取
│   │   ├── prompts.py      # 組裝 Baseline / Skill 的 system prompt
│   │   ├── llm.py          # Claude 呼叫（AsyncAnthropic + streaming）
│   │   ├── sessions.py     # 記憶體內的工作階段（兩個分支獨立）
│   │   └── server.py       # REST API + SSE
│   ├── prompts/            # baseline.md、output_format.md、各 Skill 的 wrapper 與 overview
│   ├── data/               # examples.json（範例題目）、materials.json（專案資料清單）
│   └── tests/              # 後端測試（使用假的 LLM，不會呼叫 API）
└── frontend/               # Next.js 16 + React 19 + TypeScript + Tailwind CSS 4
    └── src/
        ├── app/            # 頁面：/、/about、/skill、/materials
        ├── components/     # 比較介面、回答面板、Markdown/KaTeX 等
        └── lib/            # API client、比較狀態管理
```

---

## 2. 第一次安裝

前後端共用 conda 環境 `orqa`。

```bash
conda activate orqa

# 後端套件
pip install -r website/backend/requirements.txt

# Node.js（裝在 orqa 環境內，不影響系統）
conda install -c conda-forge nodejs

# 前端套件
cd website/frontend
npm install
```

### API Key

後端會依序讀取 `website/backend/.env`、專案根目錄的 `.env`。根目錄 `.env` 已經有 `ANTHROPIC_API_KEY` 的話，不需要再設定。
若要另外設定，複製 `website/backend/.env.example` 為 `website/backend/.env` 並填入。

API Key 只存在後端，不會傳到瀏覽器，也不會出現在任何 API 回應中。`.env` 已列在 `.gitignore`。

---

## 3. 啟動

需要開兩個終端機，兩邊都先 `conda activate orqa`。

**終端機 1：後端（http://localhost:8000）**

```bash
cd website/backend
uvicorn main:app --reload --port 8000
```

**終端機 2：前端（http://localhost:3000）**

```bash
cd website/frontend
npm run dev
```

用瀏覽器開啟 http://localhost:3000 。右上角的綠點表示已連上後端；若後端沒有啟動，首頁會顯示提示。

> 對話只保存在後端記憶體中。後端重新啟動（包含 `--reload` 偵測到程式碼變更）或閒置超過 `session_ttl_minutes` 後會清除。

---

## 4. 常見修改

| 想做的事 | 修改位置 |
|---|---|
| 換模型、effort、輸出上限、逾時 | `backend/config.yaml` 的 `llm` 區塊 |
| **Skill 在網頁上顯示的名稱** | `backend/config.yaml` 中該 Skill 的 `label`（下拉選單、問題卡片、欄位副標題、Skill 設計頁分頁都會跟著改）；說明文字在 `description` |
| **新增 / 替換可選的 Skill** | `backend/config.yaml` 的 `prompts.skills`（每筆有 `id`、`label`、`description`、`source`、`wrapper`、`overview`；`source` 可指向 `.skill` / `.zip` / 資料夾 / 單一 `.md`），預設選項為 `prompts.default_skill` |
| 調整 Skill 在網頁中的使用方式（輸出段落等） | 該 Skill 的 `wrapper`：`backend/prompts/skill_wrapper.md`（Concept Identification）、`backend/prompts/skill_wrapper_complexlp.md`（Practical Implementation）、`backend/prompts/skill_wrapper_general.md`（General Optimization） |
| 調整 Baseline 提示詞 | `backend/prompts/baseline.md` |
| 兩邊共用的格式規則（語言、Markdown、LaTeX） | `backend/prompts/output_format.md` |
| 範例題目 | `backend/data/examples.json` |
| 專案資料（海報、簡報、報告） | 把檔案放進 `assets/`，在 `backend/data/materials.json` 設定路徑 |
| 「Skill 設計」頁面內容 | 該 Skill 的 `overview`：`backend/prompts/skill_overview.md`、`backend/prompts/skill_overview_complexlp.md`、`backend/prompts/skill_overview_general.md` |

修改後端設定或 prompt 後需要重新啟動後端（或等 `--reload` 自動重啟）。

> 專題報告目前尚未放入：把 PDF 存成 `assets/專題報告.pdf` 就會自動出現在「Project Materials」頁面。

### 關於 temperature

目前的 Claude API 版本會拒絕 `temperature` / `top_p` / `top_k`，因此兩邊都使用 API 預設值；公平性由「相同模型、相同 `effort` 與 `max_tokens`、相同輸出格式規則」來保證。

---

## 5. 運作方式

```text
瀏覽器 ──POST /api/sessions──────────────────────────► 建立工作階段
       ──POST …/branches/baseline/messages ─┐  同時發出
       ──POST …/branches/skill/messages ────┘  各自以 SSE 串流回傳
                                               ├─ thinking：思考摘要
                                               ├─ delta   ：回答片段
                                               └─ done / error
```

- 兩個分支是兩個獨立請求，先完成的先顯示；一邊失敗不影響另一邊。
- 追問只送到指定的分支，只使用該分支自己的歷史。
- 只有成功的回合才寫入歷史；失敗後按「重試」只重送失敗的那一邊。
- 模型產生的程式碼只以文字顯示，不會被執行。

### API

| Method | Endpoint | 說明 |
|---|---|---|
| GET | `/api/health` | 後端狀態 |
| GET | `/api/config` | 公開設定（模型、輸入上限、可選的 Skill 清單等） |
| GET | `/api/examples` | 範例題目 |
| GET | `/api/skill?skill_id=` | 指定 Skill 的名稱、檔案與說明（省略時為預設 Skill） |
| GET | `/api/materials` | 專案資料清單 |
| GET | `/api/materials/{id}/file` | 開啟檔案（`?download=1` 下載） |
| POST | `/api/sessions` | 建立比較工作階段（body 可帶 `{"skill_id": ...}` 指定 OptiSkill 端的 Skill，整個工作階段固定使用） |
| GET | `/api/sessions/{id}` | 取得工作階段狀態 |
| DELETE | `/api/sessions/{id}` | 刪除工作階段 |
| POST | `/api/sessions/{id}/branches/{baseline\|skill}/messages` | 送出訊息（SSE 串流） |
| POST | `/api/sessions/{id}/branches/{baseline\|skill}/retry` | 重試失敗的分支（SSE 串流） |

FastAPI 自動產生的文件：http://localhost:8000/docs

---

## 6. 測試

```bash
# 後端（不會呼叫 Claude API）
cd website/backend
python -m pytest

# 前端：型別檢查、lint、production build
cd website/frontend
npx tsc --noEmit
npm run lint
npm run build
```

---

## 7. 疑難排解

| 狀況 | 處理 |
|---|---|
| 首頁顯示「無法連線到後端」 | 確認終端機 1 的 uvicorn 正在執行，且埠號是 8000 |
| 顯示「API Key 無效或未設定」 | 檢查 `.env` 的 `ANTHROPIC_API_KEY`，修改後重新啟動後端 |
| 顯示「工作階段不存在或已過期」 | 後端重新啟動過，按「新的比較」重新開始 |
| `npm` / `node` 找不到 | 先 `conda activate orqa`（Node.js 安裝在這個環境中） |
| 某一邊顯示 Timeout | 調高 `config.yaml` 的 `llm.branch_timeout`，或降低 `effort` |
| 回答被安全機制拒絕 | 調整問題措辭；`llm.refusal_fallback: true` 會自動改用相容模型重跑部分類別 |
