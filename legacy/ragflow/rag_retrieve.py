"""
    獨立測試 RAGFlow Retrieval 是否正常。
"""


from config import create_config
from ragflow_client import create_ragflow_client


TEST_QUESTION = ("What is linear programming?")

TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.2  # 這是最低相似度門檻
VECTOR_SIMILARITY_WEIGHT = 0.3
# 1.0：完全偏向 vector similarity
# 0.0：不使用 vector similarity

RETRIEVAL_TOP_K = 1024  # retrieval candidate pool
KEYWORD_SEARCH = False # 目前先使用 RAGFlow 預設 retrieval configuration

def print_retrieval_results( question: str, chunks):
    
    print()
    print("=" * 30)
    print("Retrieval Results")
    print("=" * 30)
    print()
    print("Question:")
    print(question)
    print()
    print(f"Retrieved chunks: {len(chunks)}")

    for rank, chunk in enumerate(chunks, start=1):

        print()
        print("-" * 30)
        print(f"Rank {rank}")
        print("-" * 30)

        print(f"Chunk ID      : {chunk.id}")
        print(f"Score         : {chunk.score}")
        print(f"Dataset ID    : {chunk.dataset_id}")
        print(f"Document ID   : {chunk.document_id}")
        print(f"Document Name : {chunk.document_name}")
        print(f"Page          : {chunk.page}")

        print()
        print("Content:")
        print("-" * 30)
        print(chunk.content)

        print("-" * 30)



def main():

    print()
    print("=" * 30)
    print("Phase 5: RAGFlow Retrieval API Test")
    print("=" * 30)

    config = create_config(
        use_skill=False,
        use_rag=True,
    )
    print("[Configuration loaded] ")
    print(f"RAGFlow Host     : {config.rag.host}")
    print(f"PDF Dataset      : {config.rag.pdf_dataset_name}")
    print(f"Markdown Dataset : {config.rag.md_dataset_name}")
    print(f"Top-K            : {TOP_K_RESULTS}")
    print(f"Similarity        : {SIMILARITY_THRESHOLD}")
    print(f"Vector Weight     : {VECTOR_SIMILARITY_WEIGHT}")

    ragflow = create_ragflow_client(config)

    print("[Testing RAGFlow connection] ")

    datasets = ragflow.test_connection()

    print("RAGFlow connection successful.")
    print( f"Visible datasets: {len(datasets)}")

    print("[Checking RAGFlow datasets]")

    pdf_dataset = (
        ragflow.get_dataset_by_name(
            config.rag.pdf_dataset_name
        )
    )
    md_dataset = (
        ragflow.get_dataset_by_name(
            config.rag.md_dataset_name
        )
    )

    if pdf_dataset is None:
        raise RuntimeError(f"PDF Dataset does not exist: {config.rag.pdf_dataset_name}")
    if md_dataset is None:
        raise RuntimeError(f"Markdown Dataset does not exist: {config.rag.md_dataset_name}")

    
    print("[Calling RAGFlow Retrieval API]")

    print("Question:")
    print(TEST_QUESTION)

    chunks = ragflow.retrieve(
        dataset_names=[
            config.rag.pdf_dataset_name,
            config.rag.md_dataset_name,
        ],
        question=TEST_QUESTION,
        document_ids=None,
        page=1,
        page_size=TOP_K_RESULTS,
        sim_threshold= SIMILARITY_THRESHOLD,
        vector_sim_weight= VECTOR_SIMILARITY_WEIGHT,
        top_k=RETRIEVAL_TOP_K,
        keyword=KEYWORD_SEARCH,
        rerank_id=None,
    )

    print("[Retrieval completed]")

    print_retrieval_results(
        question=TEST_QUESTION,
        chunks=chunks,
    )

    print()
    print("=" * 30)
    print("Phase 5 Verification")
    print("=" * 30)

    if not chunks:
        print("PHASE 5 FAILED" )
        print()
        print( "RAGFlow Retrieval returned no chunks.")
        
        raise RuntimeError("Phase 5 retrieval returned no results.")

    non_empty_chunks = [
        chunk
        for chunk in chunks
        if chunk.content.strip()
    ]

    if not non_empty_chunks:
        raise RuntimeError("RAGFlow returned chunks, but all chunk contents are empty.")

    invalid_ids = [
        chunk
        for chunk in chunks
        if not chunk.id
    ]
    if invalid_ids:
        raise RuntimeError("Some retrieved chunks do not have valid chunk IDs.")

    print("PHASE 5 PASSED")
    print(f"RAGFlow returned {len(chunks)} retrieved chunks.")
    
if __name__ == "__main__":
    main()