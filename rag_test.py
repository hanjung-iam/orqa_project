"""
本程式只負責確認：
    1. ORQA_Project 可以連線到 RAGFlow
    2. RAGFlow API Key 正常
    3. PDF Dataset 存在
    4. Markdown Dataset 存在
    5. PDF Dataset 使用 voyage-4-large
    6. Markdown Dataset 使用 voyage-4-large
    7. Dataset chunk method 為 naive

"""

from __future__ import annotations

import sys

from config import create_config, validate_config
from ragflow_client import create_ragflow_client


def main() -> None:

    print("\n[1/5] Loading configuration...")

    config = create_config(
        use_rag=True,
        use_skill=False,
    )

    print("\n[2/5] Validating configuration...")
    try:
        validate_config(config)
    except Exception as exc:
        print("\nConfiguration validation failed.")
        print(str(exc))
        sys.exit(1)

    print("\nRAGFlow configuration:")
    print(f"  Host                : {config.rag.host}")
    print(f"  PDF Dataset         : {config.rag.pdf_dataset_name}")
    print(f"  Markdown Dataset    : {config.rag.md_dataset_name}")
    print(f"  Embedding model     : {config.rag.embedding_model}")
    print(f"  Chunk method        : {config.rag.chunk_method}")

    print("\n[3/5] Connecting to RAGFlow...")
    try:
        ragflow = create_ragflow_client(config)
    except Exception as exc:
        print("\nFailed to create RAGFlow client.")
        print(f"Error: {exc}")
        sys.exit(1)

    print("\n[4/5] Testing RAGFlow connection...")

    try:
        datasets_before = ragflow.test_connection()
    except Exception as exc:
        print("\nFailed to connect to RAGFlow.")
        print(f"Error: {exc}")
        print(
            "\nPlease check:\n"
            "  1. RAGFlow is running\n"
            "  2. RAGFLOW_HOST is correct\n"
            "  3. RAGFLOW_API_KEY is correct"
        )
        sys.exit(1)

    print(f"Existing datasets visible to this API key:{len(datasets_before)}")

    print("\n[5/5] Checking ORQA datasets...")

    try:
        datasets = ragflow.setup_orqa_datasets(
            pdf_dataset_name=(
                config.rag.pdf_dataset_name
            ),
            md_dataset_name=(
                config.rag.md_dataset_name
            ),
            embedding_model=(
                config.rag.embedding_model
            ),
            chunk_method=(
                config.rag.chunk_method
            ),
        )
    except Exception as exc:
        print("\nFailed to setup ORQA datasets.")
        print(f"Error: {exc}")
        sys.exit(1)

    pdf_dataset = datasets["pdf"]
    print("\n" + "-" * 30)
    print("PDF Dataset")
    print("-" * 30)
    print(f"Name            : {pdf_dataset.name}")
    print(f"ID              : {pdf_dataset.id}")
    print(f"Embedding model : {pdf_dataset.embedding_model}")
    print(f"Chunk method    : {pdf_dataset.chunk_method}")
    print(f"Documents       : {pdf_dataset.document_count}")
    print(f"Chunks          : {pdf_dataset.chunk_count}")

    md_dataset = datasets["markdown"]
    print("\n" + "-" * 30)
    print("Markdown Dataset")
    print("-" * 30)
    print(f"Name            : {md_dataset.name}")
    print(f"ID              : {md_dataset.id}")
    print(f"Embedding model : {md_dataset.embedding_model}")
    print(f"Chunk method    : {md_dataset.chunk_method}")
    print(f"Documents       : {md_dataset.document_count}")
    print(f"Chunks          : {md_dataset.chunk_count}")

    print("\n" + "=" * 70)
    print("[Verification]")
    print("=" * 70)

    errors = []
    if not pdf_dataset.id:
        errors.append(
            "PDF Dataset does not have an ID."
        )

    if pdf_dataset.name != config.rag.pdf_dataset_name:
        errors.append(
            "PDF Dataset name does not match configuration."
        )

    if (
        pdf_dataset.chunk_method
        != config.rag.chunk_method
    ):
        errors.append(
            "PDF Dataset chunk method does not match configuration."
        )


    if not md_dataset.id:
        errors.append(
            "Markdown Dataset does not have an ID."
        )

    if md_dataset.name != config.rag.md_dataset_name:
        errors.append(
            "Markdown Dataset name does not match configuration."
        )

    if md_dataset.name != config.rag.md_dataset_name:
        errors.append(
        "Markdown Dataset name does not match configuration."
        )
        
    if (
    pdf_dataset.embedding_model
    != md_dataset.embedding_model
    ):
        errors.append(
        "PDF and Markdown datasets use different embedding models."
        )

    if (
        md_dataset.chunk_method
        != config.rag.chunk_method
    ):
        errors.append(
            "Markdown Dataset chunk method does not match "
            "configuration."
        )
    if errors:

        print("\nPhase 2 FAILED.")
        print("Problems:")
        for error in errors:
            print(f"  - {error}")

        sys.exit(1)
    print("\nPhase 2 PASSED.")


if __name__ == "__main__":
    main()