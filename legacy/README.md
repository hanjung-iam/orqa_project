# legacy/

此目錄保存已不再維護、但保留作為參考的舊程式。這些檔案**不會被新版 pipeline import**，也不保證可以直接執行。

| 檔案 | 說明 |
|---|---|
| `test_dify_rag.py` | 舊版 Dify Chatflow RAG 測試腳本（結果位於 `results/dify_rag*`） |
| `rag_colab_reference.py` | Colab 版 RAG 參考程式（Chunking / Voyage / Chroma / BM25 / RRF 的原始實作），新版 `orqa/rag/` 依此移植 |
| `ragflow/` | 舊版 RAGFlow 整合（`ragflow_client.py`、`rag_test/upload/parse/retrieve.py`）。新版已改用自建 RAG（`orqa/rag/`），不再依賴 RAGFlow |
| `v1/` | 重構前的主程式（`main.py`、`inference.py`、`llm.py`、`prompt.py`、`storage.py` ...），由 `orqa/` 套件取代。舊結果（`results/0`、`Basic`、`skill`、`dify_rag*`）保留在原位置，不轉換 |
