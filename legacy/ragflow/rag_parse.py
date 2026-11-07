"""
本程式負責：

1. 讀取 ORQA 專案設定
2. 連線 RAGFlow
3. 找到 Phase 3 已上傳的 PDF 文件
4. 找到 Phase 3 已上傳的 Markdown 文件
5. 啟動 RAGFlow Document Parsing
6. 持續監控 Parsing progress
7. 等待所有文件完成
8. 確認文件狀態為 DONE
9. 確認已產生 chunks
10. 印出 Phase 4 驗證結果

"""

from pathlib import Path

from config import create_config
from ragflow_client import create_ragflow_client


def print_document_info(document, title: str):

    print(title)

    print(f"Document ID   : {document.id}")
    print(f"Document Name : {document.name}")
    print(f"Dataset ID    : {document.dataset_id}")
    print(f"Run Status    : {document.run}")
    print(f"Chunk Count   : {document.chunk_count}")


def main():
    print()
    print("=" * 70)
    print("ORQA - Phase 4: RAGFlow Document Parsing")
    print("=" * 70)

    config = create_config(
        use_skill=False,
        use_rag=True,
    )

    print("[1/6] Configuration loaded.")
    print(f"RAGFlow Host       : {config.rag.host}")

    print(f"PDF Dataset        : {config.rag.pdf_dataset_name}")
    print(f"Markdown Dataset   : {config.rag.md_dataset_name}")
    print(f"Embedding Model    : {config.rag.embedding_model}")
    print(f"Chunk Method       : {config.rag.chunk_method}")

    ragflow = create_ragflow_client(config)

    print("[2/6] Testing RAGFlow connection...")

    datasets = ragflow.test_connection()

    print(f"RAGFlow connection successful.")
    print(f"Visible datasets: {len(datasets)}")

    print("[3/6] Checking ORQA datasets...")

    pdf_dataset = ragflow.get_dataset_by_name(
        config.rag.pdf_dataset_name
    )

    md_dataset = ragflow.get_dataset_by_name(
        config.rag.md_dataset_name
    )

    if pdf_dataset is None:
        raise RuntimeError(f"PDF Dataset does not exist: {config.rag.pdf_dataset_name}")

    if md_dataset is None:
        raise RuntimeError(f"Markdown Dataset does not exist: {config.rag.md_dataset_name}")

    print(f"PDF Dataset found: {config.rag.pdf_dataset_name}")
    print(f"Markdown Dataset found: {config.rag.md_dataset_name}")

    print("[4/6] Finding uploaded documents...")

    pdf_documents = ragflow.get_documents(
        config.rag.pdf_dataset_name
    )
    if not pdf_documents:
        raise RuntimeError("No documents found in PDF Dataset.")
    
    print("PDF Dataset documents:")
    for document in pdf_documents:

        print(
            f"  - {document.name}"
            f" | id={document.id}"
            f" | run={document.run}"
            #f" | progress={document.progress}"
            f" | chunks={document.chunk_count}"
        )

    md_documents = ragflow.get_documents(
        config.rag.md_dataset_name
    )

    if not md_documents:
        raise RuntimeError("No documents found in Markdown Dataset.")

    print("Markdown Dataset documents:")

    for document in md_documents:
        print(
            f"  - {document.name}"
            f" | id={document.id}"
            f" | run={document.run}"
            #f" | progress={document.progress}"
            f" | chunks={document.chunk_count}"
        )

    pdf_document = pdf_documents[0]
    md_document = md_documents[0]

    print_document_info(
        pdf_document,
        "PDF document before parsing",
    )

    print_document_info(
        md_document,
        "Markdown document before parsing",
    )

    print("[5/6] Starting RAGFlow Document Parsing...")

    print("Starting PDF parsing...")

    ragflow.async_parse_documents(
        dataset_name=config.rag.pdf_dataset_name,
        document_ids=[pdf_document.id],
    )

    print(f"PDF parsing started: {pdf_document.name}")

    print("Starting Markdown parsing...")

    ragflow.async_parse_documents(
        dataset_name=config.rag.md_dataset_name,
        document_ids=[md_document.id],
    )

    print(f"Markdown parsing started: {md_document.name}")

    print("[6/6] Waiting for RAGFlow parsing to finish...")

    print()
    print("=" * 70)
    print("PDF Parsing")
    print("=" * 70)

    pdf_result = ragflow.wait_for_parsing(
        dataset_name=config.rag.pdf_dataset_name,
        document_ids=[pdf_document.id],
        poll_interval=2,
        timeout=3600,
    )

    print()
    print("=" * 70)
    print("Markdown Parsing")
    print("=" * 70)

    md_result = ragflow.wait_for_parsing(
        dataset_name=config.rag.md_dataset_name,
        document_ids=[md_document.id],
        poll_interval=2,
        timeout=3600,
    )

    print("=" * 70)
    print("Phase 4 Verification")
    print("=" * 70)

    errors = []

    pdf_final = pdf_result[0]

    print("[PDF]")
    print(f"Document : {pdf_final.name}")
    print(f"Run      : {pdf_final.run}")
    #print(f"Progress : {pdf_final.progress}")
    print(f"Chunks   : {pdf_final.chunk_count}")

    if str(pdf_final.run).upper() != "DONE":
        errors.append("PDF document parsing did not finish with DONE.")

    if (
        pdf_final.chunk_count is None
        or pdf_final.chunk_count <= 0
    ):
        errors.append("PDF document produced no chunks.")

    md_final = md_result[0]

    print("[Markdown]")
    print(f"Document : {md_final.name}")
    print(f"Run      : {md_final.run}")
    #print(f"Progress : {md_final.progress}")
    print(f"Chunks   : {md_final.chunk_count}")

    if str(md_final.run).upper() != "DONE":

        errors.append("Markdown document parsing did not finish with DONE.")

    if (
        md_final.chunk_count is None
        or md_final.chunk_count <= 0
    ):
        errors.append("Markdown document produced no chunks.")

    print()
    print("=" * 70)

    if errors:
        print("PHASE 4 FAILED")
        print("=" * 70)
        print()
        for error in errors:
            print(f"[ERROR] {error}")

        raise RuntimeError("Phase 4 verification failed.")

    print("PHASE 4 PASSED")
    print("=" * 70)

    print()
    print("RAGFlow Document Parsing completed successfully.")

    print()
    print("PDF:")
    print(
        f"  {pdf_final.name} "
        f"→ DONE "
        f"→ {pdf_final.chunk_count} chunks"
    )

    print()
    print("Markdown:")
    print(
        f"  {md_final.name} "
        f"→ DONE "
        f"→ {md_final.chunk_count} chunks"
    )

    print()
    print("The documents are now ready for the next RAG retrieval phase.")

if __name__ == "__main__":
    main()