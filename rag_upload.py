from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    create_config,
    validate_config,
)

from ragflow_client import (
    create_ragflow_client,
)

def parse_arguments():
    """
        python rag/upload_phase3.py \
            --pdf "data/OR textbook.pdf" \
            --markdown "data/structured.md"
    """
    parser = argparse.ArgumentParser(
        description=(
            "Upload ORQA textbook materials to RAGFlow."
        )
    )

    parser.add_argument(
        "--pdf", required=True, type=str, help=("Path to OR textbook.pdf"),
    )

    parser.add_argument(
        "--markdown", required=True, type=str, help=("Path to structured.md"),
    )

    return parser.parse_args()

def main() -> None:

    args = parse_arguments()

    pdf_path = Path(args.pdf).resolve()

    markdown_path = Path(args.markdown).resolve()

    print("\n[Input files]")

    print(f"PDF       : {pdf_path}")
    print(f"Markdown  : {markdown_path}")

    print("\n[1/5] Checking local files...")

    if not pdf_path.exists():
        print(f"ERROR: PDF file does not exist:\n{pdf_path}")
        sys.exit(1)

    if not pdf_path.is_file():
        print(f"ERROR: PDF path is not a file:\n {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: Expected a PDF file, but got: {pdf_path.suffix}"
        )
        sys.exit(1)

    if not markdown_path.exists():
        print(f"ERROR: Markdown file does not exist:\n{markdown_path}")
        sys.exit(1)

    if not markdown_path.is_file():
        print(f"ERROR: Markdown path is not a file:\n{markdown_path}")
        sys.exit(1)

    if markdown_path.suffix.lower() != ".md":
        print(f"ERROR: Expected a Markdown file, but got: {markdown_path.suffix}")
        sys.exit(1)

    print("Local files are valid.")

    print("\n[2/5] Loading RAGFlow configuration...")

    config = create_config(
        use_rag=True,
        use_skill=False,
    )

    try:
        validate_config(config)

    except Exception as exc:

        print( "\nConfiguration validation failed.")
        print(f"Error: {exc}")
        sys.exit(1)


    print("\n[3/5] Connecting to RAGFlow...")

    try:
        ragflow = (create_ragflow_client(config))
        ragflow.test_connection()

    except Exception as exc:
        print("\nFailed to connect to RAGFlow.")
        print(f"Error: {exc}")
        sys.exit(1)

    print("RAGFlow connection successful.")


    print("\n[4/5] Uploading OR textbook.pdf...")

    print(f"Target Dataset: {config.rag.pdf_dataset_name}")

    try:
        pdf_document = (
            ragflow.upload_file(
                dataset_name=(config.rag.pdf_dataset_name),
                file_path=pdf_path,
            )
        )
    except Exception as exc:

        print("\nFailed to upload PDF.")
        print(f"Error: {exc}")
        sys.exit(1)

    print("PDF uploaded successfully.")
    print(f"  Document ID : {pdf_document.id}")
    print(f"  Name        : {pdf_document.name}")
    print(f"  Dataset ID  : {pdf_document.dataset_id}")
    print(f"  Run status  : {pdf_document.run}")
    print(f"  Chunk count : {pdf_document.chunk_count}")

    print("\n[5/5] Uploading structured.md...")

    print( f"Target Dataset: {config.rag.md_dataset_name}")

    try:
        markdown_document = (
            ragflow.upload_file(
                dataset_name=(config.rag.md_dataset_name),
                file_path=markdown_path,
            )
        )

    except Exception as exc:
        print("\nFailed to upload Markdown.")
        print(f"Error: {exc}")
        sys.exit(1)

    print("Markdown uploaded successfully.")
    print(f"  Document ID : {markdown_document.id}")
    print(f"  Name        : {markdown_document.name}")
    print(f"  Dataset ID  : {markdown_document.dataset_id}")
    print(f"  Run status  : {markdown_document.run}")
    print(f"  Chunk count : {markdown_document.chunk_count}")


if __name__ == "__main__":
    main()