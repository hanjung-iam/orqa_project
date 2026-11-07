# ORQA Experiment Framework (v2)

ORQA（Operations Research Question Answering）四選一問答的實驗框架，用來比較不同 LLM、Prompt、Skill 與 RAG 方法的表現。

- **LLM Provider**：Anthropic Claude API／Hugging Face Transformers（本機 GPU、4-bit 量化），共用同一個 Inference Engine
- **推論模式**：`basic`、`skill`、`rag`、`skill_rag`
- **RAG**：Voyage Embedding + 本機 ChromaDB + BM25 + Reciprocal Rank Fusion（不再依賴 RAGFlow）
- **結果**：原始預測（Raw）與 Haiku 後處理（Post-analysis）完全分開，各自計算準確率
- **實驗管理**：每次實驗一個獨立目錄，支援中斷續跑，絕不覆寫舊結果

---

## 1. 專案結構

```text
.
├── main.py                     # 實驗入口（推論 → 後處理 → 評估）
├── config/experiment.yaml      # 所有實驗參數
├── orqa/
│   ├── config.py               # YAML 載入與驗證；API Key 只從環境變數讀取
│   ├── dataset.py              # ORQA 資料集（index = 0-based 行號，記錄 SHA256）
│   ├── skill.py                # Skill 載入（ZIP / .skill / 目錄），固定順序 + hash
│   ├── prompts.py              # Prompt Builder（prompt v2）
│   ├── parser.py               # 答案解析（strict + 寬鬆規則），Invalid = null
│   ├── llm/
│   │   ├── base.py             # BaseLLMProvider 介面
│   │   ├── anthropic_provider.py
│   │   └── transformers_provider.py
│   ├── rag/
│   │   ├── documents.py        # 知識庫文件載入與版本
│   │   ├── chunking.py         # Heading-based / Case-aware Chunking
│   │   ├── embedding.py        # Voyage Embedding
│   │   ├── vector_store.py     # ChromaDB
│   │   ├── retrieval.py        # Dense / BM25 / RRF
│   │   ├── pipeline.py         # Index 建立與檢索
│   │   └── __main__.py         # python -m orqa.rag ...
│   ├── engine.py               # 單一 Inference Engine
│   ├── analysis.py             # Haiku Invalid Answer 後處理
│   ├── evaluation.py           # Raw / Post-analysis Evaluation
│   ├── storage.py              # 結果檔案讀寫（append-only / 原子寫入）
│   └── experiment.py           # 實驗目錄與 experiment.json
├── dataset/                    # ORQA_test.jsonl（1468 題）、ORQA_validation.jsonl
├── skill/                      # Skill 檔案（.skill = ZIP）
├── data/                       # （不進 git）knowledge_base/、vector_db/
├── results/                    # （不進 git）每個實驗一個子目錄；舊版結果保留在原位置
├── tests/                      # unit / integration 測試
└── legacy/                     # 舊版程式（v1、RAGFlow、Dify、Colab RAG 參考程式），不再維護
```

---

## 2. 安裝

### 2.1 Python 環境

建議使用 **Python 3.12**（`chromadb` 依賴的 `onnxruntime` 目前沒有 Python 3.15 的 wheel）。

```bash
conda create -n orqa-v2 python=3.12 -y
conda activate orqa-v2
pip install -r requirements.txt
```

### 2.2 本機模型（只在 GPU 機器）

```bash
# 依驅動支援的 CUDA 版本安裝 torch（範例為 CUDA 12.8）
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt
```

### 2.3 API Key

複製 `.env.example` 為 `.env` 並填入：

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
HF_TOKEN=            # 選填
```

`.env` 已在 `.gitignore` 中。程式只會從環境變數讀取 Key，不會寫入任何結果檔或 log。

> 舊版 `.env` 中的 `RAGFLOW_*`、`RAG_*`、`LLM_*`、`REPAIR_*` 等變數已不再使用，實驗參數改由 `config/experiment.yaml` 控制。

---

## 3. 設定檔

所有參數集中在 `config/experiment.yaml`（欄位寫錯或型別錯誤會在啟動時直接報錯）。常用欄位：

| 區塊 | 欄位 | 說明 |
|---|---|---|
| `experiment` | `method` | `basic` / `skill` / `rag` / `skill_rag` |
| | `limit` | 只跑前 N 題（smoke test） |
| `llm` | `provider` | `anthropic` / `transformers` |
| `llm.anthropic` | `model_id`、`max_tokens`、`extra_params` | Claude 設定 |
| `llm.transformers` | `model_id`、`quantization`、`batch_size`、`max_new_tokens`、`enable_thinking` | 本機模型設定 |
| `post_analysis` | `enabled`、`model_id` | Haiku 後處理（預設 `claude-haiku-4-5-20251001`） |
| `rag.documents` | `path`、`strategy` | 知識庫文件與 chunking 策略 |
| `rag.retrieval` | `mode`、`similarity_threshold`、`dense_top_k`、`bm25_top_k`、`final_top_k`、`rrf_k` | 檢索參數 |

CLI 可覆寫部分設定：`--method`、`--provider`、`--model-id`、`--limit`、`--experiment-id`。

> **Claude 的 sampling 參數**：`anthropic` SDK 2.x 使用 API 版本 `2026-10-21`，該版本對 `temperature` / `top_p` / `top_k` 一律回傳 400，因此 Claude 一律使用 API 預設值，設定檔也會拒絕這些欄位。（舊版程式雖然在設定中寫了 `temperature: 0.0`，但實際上從未傳給 API。）

---

## 4. 本機模型（Hugging Face Transformers）

### 4.1 預設模型：Qwen

預設模型為 **`Qwen/Qwen3.5-9B`**（Apache-2.0、9.65B 參數、context 262K），4-bit NF4 量化後估計約需 6–8 GB VRAM（含 KV cache），應可在 12 GB 上執行。較小的 baseline 可用 `Qwen/Qwen3.5-4B`。

```bash
huggingface-cli download Qwen/Qwen3.5-9B
python -m pytest tests/integration/test_transformers_provider.py -m gpu -v -s      # Provider 測試
python main.py --provider transformers --method basic --limit 5                      # 5 題 smoke test
```

- Qwen3.5 之後的官方 repo 沒有 `-Instruct` 後綴；沒有 `-Base` 的 repo 就是對話模型。
- Qwen3.5 預設會先輸出思考內容（thinking）。主實驗設定 `enable_thinking: false`；若改為 `true`，請一併調高 `max_new_tokens`，否則思考段落被截斷會造成大量 Invalid。
- 27B（Qwen3.5 / 3.6 / 3.8）4-bit 約需 15 GB 以上，超過 12 GB，需要 CPU offload，速度會大幅下降，目前不使用。

### 4.2 改用其他開源模型（以 Gemma 4 為例）

任何能用 Hugging Face Transformers 載入、且有 chat template 的對話模型都可以使用，**不需要修改程式碼**，只要更換 `model_id`。以下以 Google Gemma 4 為例。

#### 步驟 1：選擇模型

在 Hugging Face 模型頁面確認：

| 檢查項目 | 說明 |
|---|---|
| 對話（Instruct）版本 | Gemma 選 `-it` 結尾；Qwen3.5 以後選不帶 `-Base` 的版本。不要用 base 模型 |
| Transformers 格式 | 不要選 `*-gguf`、`*-ct` 等其他推論框架專用的 repo |
| 授權與 gated | Gemma 4 為 Apache-2.0、不需申請。若模型是 gated（頁面要求同意授權），先在網頁上同意，並在 `.env` 設定 `HF_TOKEN` |
| Transformers 版本 | 看 repo 中 `config.json` 的 `transformers_version`。Gemma 4 需要 ≥ 5.10（`requirements-gpu.txt` 為 5.15.1） |
| VRAM 估算 | 4-bit 約為「參數量（B）× 0.5 GB」，再加 1–2 GB 給 KV cache 等 |

Gemma 4 可選的版本（VRAM 為估算值，尚未實測）：

| Model ID | 參數量 | 4-bit 估計 VRAM | 12 GB 可行性 |
|---|---|---|---|
| `google/gemma-4-E4B-it` | 約 8.0B（含 per-layer embedding，有效約 4B） | 約 5–6 GB | ✅ 輕量 baseline |
| `google/gemma-4-12B-it` | 約 12.0B | 約 7–9 GB | ✅ 建議 |
| `google/gemma-4-26B-A4B-it` | 約 25.8B（MoE） | 約 14–16 GB | ❌ 需要 CPU offload |
| `google/gemma-4-31B-it` | 約 31.3B | 約 17–19 GB | ❌ 需要 CPU offload |

#### 步驟 2：下載模型

```bash
huggingface-cli download google/gemma-4-12B-it
```

（也可以略過這一步，第一次執行時會自動下載。）

#### 步驟 3：切換模型（三種方式擇一）

**A. CLI 單次覆寫**（最快，適合 smoke test）

```bash
python main.py --provider transformers --model-id google/gemma-4-12B-it --method basic --limit 5
```

**B. 修改 `config/experiment.yaml`**

```yaml
llm:
  provider: transformers
  transformers:
    model_id: google/gemma-4-12B-it
```

**C. 為每個模型建立獨立設定檔**（建議用於正式比較實驗）

複製 `config/experiment.yaml` 為 `config/gemma4_12b.yaml`，只修改 `llm` 區塊，其他設定（資料集、Skill、RAG、Prompt）保持相同，確保比較公平：

```yaml
llm:
  provider: transformers
  transformers:
    model_id: google/gemma-4-12B-it
    quantization: 4bit
    max_new_tokens: 512
    batch_size: 1
    enable_thinking: false
```

```bash
python main.py --config config/gemma4_12b.yaml --method skill_rag
```

每個實驗都有獨立的 `results/<experiment_id>/`（ID 含模型名稱），不同模型的結果不會互相覆蓋；`python main.py --list` 可一起列出比較。

#### 步驟 4：驗證

```bash
# Provider 整合測試（指定要測的模型）
ORQA_TEST_HF_MODEL=google/gemma-4-12B-it python -m pytest tests/integration/test_transformers_provider.py -m gpu -v -s
# Windows PowerShell：
#   $env:ORQA_TEST_HF_MODEL="google/gemma-4-12B-it"; python -m pytest tests/integration/test_transformers_provider.py -m gpu -v -s

# 5 題 smoke test
python main.py --provider transformers --model-id google/gemma-4-12B-it --method basic --limit 5
```

跑完後檢查：

- `experiment.json` 的 `provider_info`：`loaded_model_class`（實際載入的模型類別）、`gpu.max_memory_allocated_gb`（實際 VRAM 用量）、`system_role_merged_into_user`
- `evaluation_raw.json` 的 `invalid_count` 與 `parse_method_counts`：Invalid 很多代表模型沒有照格式輸出，可檢查 `predictions_raw.jsonl` 的 `raw_response` 與 `stop_reason`（`max_new_tokens` 表示輸出被截斷）

#### 注意事項

- **Prompt 一致**：所有模型使用相同的 System / User Prompt，只有 chat template 的格式不同。若模型的 chat template 不接受 `system` 角色（例如舊版 Gemma），程式會自動把 System Prompt 放在 user 訊息最前面（文字內容不變），並在 `experiment.json` 記錄 `system_role_merged_into_user: true`。
- **多模態架構**：Gemma 4 與 Qwen3.5 都是多模態架構。程式先用 `AutoModelForCausalLM` 載入，失敗時自動改用 `AutoModelForImageTextToText`；也可以在設定中直接指定 `auto_model_class: image_text_to_text`。
- **Thinking**：`enable_thinking` 只對支援此選項的 chat template（例如 Qwen3.5）有效，其他模型會忽略。
- **VRAM 不足（CUDA out of memory）**：程式會自動把 batch 對半拆分重試；仍不足時可降低 `batch_size`、`max_new_tokens`，或改用較小的模型。
- Gemma 4 的設定與 VRAM 數字尚未在 GPU 機器上實測，第一次使用請先跑步驟 4 的測試。

---

## 5. RAG

### 5.1 準備知識庫

將文件放在 `data/knowledge_base/`（此目錄不進 git）：

| 檔案 | Chunking 策略 |
|---|---|
| `StructuredOR.md` | `case_aware`：以 `## CASE <id>` 為邊界，同一 case 的 `###` section 優先放在同一個 chunk |
| `version_0.md` | `heading_based`：依 Markdown heading 階層切分，`heading_path` 保留完整路徑 |

`chunk_size = 2048`、`stride = 256`（超長段落以固定長度切分，相鄰片段重疊 256 字元）。Chunking 邏輯移植自 `legacy/rag_colab_reference.py`，並有 golden test 驗證 chunk 文字與舊版逐字相同（目前知識庫產生 2,995 個 chunk：StructuredOR 319、version_0 2,676）。

### 5.2 建立 Index

```bash
python -m orqa.rag stats  --config config/experiment.yaml    # chunk 統計（不呼叫 API）
python -m orqa.rag build  --config config/experiment.yaml    # Voyage embedding → ChromaDB
python -m orqa.rag status --config config/experiment.yaml
python -m orqa.rag list   --config config/experiment.yaml    # 列出所有 collection
```

- Collection 名稱為 `orqa_kb_<index_key>`。`index_key` 由文件內容 hash、chunking 設定與 embedding model 計算，任何一項改變都會建立新的 collection，不會混用不同版本的 embedding；舊 collection 保留。
- 建立過程中斷後重新執行 `build`，只會補 embedding 缺少的 chunk。
- `build --force` 只會刪除「目前設定對應的」collection 後重建。

### 5.3 單獨測試 Retrieval

```bash
python -m orqa.rag retrieve --config config/experiment.yaml --limit 5
python -m orqa.rag retrieve --config config/experiment.yaml --limit 5 --mode bm25   # 不需要 Voyage
python -m orqa.rag retrieve --config config/experiment.yaml --query "What is a binary variable?"
```

結果寫入 `results/retrieval_runs/<timestamp>_<mode>.jsonl`，每筆包含 chunk_id、dense rank / similarity / distance、BM25 rank / score、RRF score 與 metadata。

### 5.4 檢索方法

- **Query**：`Context + Question`（不包含選項，避免干擾選項影響檢索）
- **Dense**：ChromaDB cosine distance，`similarity = 1 − distance`；`similarity < 0.5` 的結果不進入 Dense 候選
- **BM25**：`rank-bm25`（BM25Okapi），tokenize = 小寫 + `\b\w+\b`；保留原始分數；不套用 similarity threshold
- **Hybrid（RRF）**：`score(d) = Σ 1 / (60 + rank(d))`，只使用排名；同時出現在兩邊的 chunk 分數相加；取前 `final_top_k = 5`
- `mode: dense` / `bm25` 可做 ablation；BM25 模式不需要 Voyage 或 ChromaDB

---

## 6. 執行實驗

```bash
# 新實驗（設定以 YAML 為準）
python main.py --config config/experiment.yaml

# 四種模式
python main.py --method basic
python main.py --method skill
python main.py --method rag            # 需先建立 RAG index
python main.py --method skill_rag

# 本機模型
python main.py --provider transformers --method skill_rag

# Smoke test（前 5 題）／只看 prompt 不呼叫 LLM
python main.py --method basic --limit 5
python main.py --method rag --dry-run
```

### 6.1 Prompt（v2）

| 模式 | System Prompt | User Prompt |
|---|---|---|
| basic | 作答規則 | 題目 |
| skill | 作答規則 + Skill 全文 | 題目 |
| rag | RAG 使用規則 + 作答規則 | Retrieved Knowledge + 題目 |
| skill_rag | 作答規則 + RAG 使用規則 + Skill 全文 | Retrieved Knowledge + 題目 |

- 「題目」= `Context / Question / A–D` + `Answer: Among A through D, the answer is (`
- Skill 一律放在 System Prompt，所有 Provider 使用相同文字（不使用 Claude Native Skill API）。
- RAG 規則明確告知模型：檢索內容不一定相關、需自行判斷、檢索內容不是答案。

### 6.2 中斷與續跑

```bash
python main.py --resume <experiment_id>                  # 從中斷處繼續
python main.py --resume <experiment_id> --limit 1468     # 把 smoke test 擴大成完整實驗
python main.py --list                                    # 列出所有實驗與準確率
```

- 以 `index` 判斷已完成的題目，不使用筆數，不會重複寫入或跳題。
- 推論失敗的題目不會寫入結果（記錄在 `inference_errors.jsonl`），續跑時會重試；連續失敗 `max_consecutive_errors` 次會自動中止。API Key 錯誤等無法重試的錯誤會立即停止。
- 續跑使用 `experiment.json` 中保存的設定，並檢查資料集、Skill、System Prompt、RAG index 是否與建立時相同，不同就拒絕續跑。

### 6.3 分階段執行

```bash
python main.py --resume <id> --stage post_analysis              # 只做 Haiku 後處理（已完成的不重做）
python main.py --resume <id> --stage post_analysis --reanalyze  # 明確要求才重新分析
python main.py --resume <id> --stage evaluate                   # 只重新計算評估
```

---

## 7. 結果檔案

每個實驗：`results/<experiment_id>/`（預設 ID：`<時間>_<method>_<model>`）

| 檔案 | 類型 | 內容 |
|---|---|---|
| `experiment.json` | metadata | 設定 snapshot、資料集 / Skill / Prompt / RAG 指紋、模型資訊、狀態、每次執行紀錄與 token 用量 |
| `predictions_raw.jsonl` | **不可變** | 原始預測（index、question_type、prediction、target_answer、raw_response、parse_method、model_id、provider、inference_mode、tokens …） |
| `invalid_answers.jsonl` | append-only | 原始輸出無法解析的題目（prediction = null） |
| `retrievals.jsonl` | append-only | RAG 模式每題的檢索結果 |
| `inference_errors.jsonl` | append-only | 推論失敗紀錄 |
| `haiku_analysis.jsonl` | append-only | Haiku 分析（resolved / unresolved / failed） |
| `predictions_post_analysis.jsonl` | 衍生 | Raw + Haiku 依 index 合併 |
| `checked_invalid.jsonl` | 衍生 | Haiku 仍無法判斷的題目 |
| `evaluation_raw.json` / `evaluation_post_analysis.json` | 衍生 | 兩套獨立評估 |
| `wrong_predictions_raw.json` / `wrong_predictions_post_analysis.json` | 衍生 | 答錯題目（含題目、選項、raw_response） |

- 原始檔只會 append（每筆 fsync），寫到一半中斷的行在讀取時會被略過；衍生檔以原子寫入產生，可隨時重建。
- 推論完成後會記錄 `predictions_raw.jsonl` 的 SHA256；post-analysis 後與評估時都會檢查它是否被修改。

### 7.1 答案解析（Parser）

依序套用以下確定性規則，`parse_method` 會記錄使用了哪一條：

1. `strict`：整段輸出就是 A/B/C/D
2. `wrapped`：`(C)`、`C)`、`**C**`、`C.`
3. `final_line`：推理後最後一行是單獨的大寫字母（例如 `**B**`）
4. `statement`：`The answer is (C)`、`Answer: C`、`Final answer: **B**`（多個語句互相矛盾時視為 Invalid）
5. `leading`：`B) Because ...`

> 用舊版 Sonnet 5.5 的 932 筆 Invalid 回答驗證：全部都能以 `final_line` 規則解析，其中 931 筆與當時 Haiku 的判斷一致。

### 7.2 Haiku 後處理

Haiku 的任務是「判斷原始模型實際選了哪個選項」，**不是重新作答**：只提供四個選項與原始回答（不提供 Context / Question），以 structured output 回傳 `A/B/C/D/UNRESOLVED`。回答沒有明確選擇、選了多個或被截斷時回傳 `UNRESOLVED`。

---

## 8. Evaluation 指標

| 指標 | 定義 |
|---|---|
| Accuracy | correct / total_questions（Invalid、Missing 都算錯） |
| Invalid Rate | invalid / total_questions |
| Q1–Q11 | 各題型 accuracy、錯誤數、`wrong_distribution`（預測選項分布、正解分布、占所有錯誤比例） |
| Haiku Resolution Rate | resolved_invalid / original_invalid |
| Checked Invalid Rate | checked_invalid / total_questions |
| Checked Invalid Ratio | checked_invalid / original_invalid |
| Accuracy Improvement | post_analysis_accuracy − raw_accuracy |

分母為 0 時回傳 `null`。

---

## 9. 測試

```bash
python -m pytest                       # unit + integration（不呼叫任何付費 API、不需要 GPU）
python -m pytest -m gpu                # Transformers Provider（GPU 機器）
```

測試涵蓋：設定驗證、資料集、Skill、Parser、Prompt、Chunking（含與舊版逐字比對）、Embedding 介面、Dense / BM25 / RRF（含與舊版比對）、ChromaDB、Index 續建、Storage、Resume、Post-analysis（raw 不可變、冪等）、兩套 Evaluation、完整 pipeline（中斷 → 續跑）。

---

## 10. 舊版結果

`results/0`、`results/Basic`、`results/skill`、`results/dify_rag*` 為舊版程式的結果，**保留在原位置、不做轉換**。注意舊版 `evaluation.json` 的準確率已混入 Haiku 修復結果（例如 `Basic` 的 832 題 Invalid 被計為有效答案），與新版的 Raw Accuracy 定義不同；`python main.py --list` 會將它們標示為 `legacy (v1)`。
