from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from ragflow_sdk import RAGFlow

@dataclass(frozen=True)
class RAGFlowDatasetInfo:
    id: str # RAGFlow Dataset ID
    name: str # Dataset name
    embedding_model: str
    chunk_method: str
    document_count: int
    chunk_count: int

class RAGFlowClient:

    def __init__(self,host: str,api_key: str):

        self.client = RAGFlow(api_key=api_key, base_url=host)

    # 測試目前是否可以透過 SDK 存取 RAGFlow
    def test_connection(self) -> list[RAGFlowDatasetInfo]:
        datasets = self.client.list_datasets(
            page=1,
            page_size=30,
        )
        return [
            self._to_dataset_info(dataset)
            for dataset in datasets
        ]


    def get_dataset_by_name(self,name: str,):
        datasets = self.client.list_datasets(
            page=1,
            page_size=30,
            name=name,
        )
        if not datasets:
            return None
        return datasets[0]

    def create_dataset(
        self,
        name: str,
        embedding_model: str,
        chunk_method: str = "naive",
        description: Optional[str] = None,
    ):
        if not name:
            raise ValueError(
                "Dataset name cannot be empty."
            )

        if not embedding_model:
            raise ValueError(
                "Embedding model cannot be empty."
            )

        if not chunk_method:
            raise ValueError(
                "Chunk method cannot be empty."
            )

        dataset = self.client.create_dataset(
            name=name,
            description=description,
            embedding_model=embedding_model,
            permission="me",
            chunk_method=chunk_method,
        )
        return dataset

    def get_or_create_dataset(
        self,
        name: str,
        embedding_model: str,
        chunk_method: str = "naive",
        description: Optional[str] = None,
    ):
        dataset = self.get_dataset_by_name(name)

        if dataset is not None:
            return dataset, False

        dataset = self.create_dataset(
            name=name,
            embedding_model=embedding_model,
            chunk_method=chunk_method,
            description=description,
        )
        return dataset, True

    def setup_orqa_datasets(
        self,
        pdf_dataset_name: str,
        md_dataset_name: str,
        embedding_model: str,
        chunk_method: str = "naive",
    ) -> dict[str, RAGFlowDatasetInfo]:
        
        pdf_dataset, pdf_created = (
            self.get_or_create_dataset(
                name=pdf_dataset_name,
                embedding_model=embedding_model,
                chunk_method=chunk_method,
                description=("ORQA knowledge base from PDF files."),
            )
        )

        md_dataset, md_created = (
            self.get_or_create_dataset(
                name=md_dataset_name,
                embedding_model=embedding_model,
                chunk_method=chunk_method,
                description=("ORQA knowledge base from Markdown files."),
            )
        )

        return {
            "pdf": self._to_dataset_info(pdf_dataset),
            "markdown": self._to_dataset_info(md_dataset),
        }

    @staticmethod
    def _to_dataset_info(dataset) -> RAGFlowDatasetInfo:

        return RAGFlowDatasetInfo(
            id=getattr(dataset, "id", ""),
            name=getattr(dataset, "name", ""),
            embedding_model=getattr(dataset,"embedding_model",None),
            chunk_method=getattr(dataset,"chunk_method",None),
            document_count=getattr(dataset,"document_count",None),
            chunk_count=getattr(dataset,"chunk_count",None),
        )


def create_ragflow_client(config) -> RAGFlowClient:

    return RAGFlowClient(
        host=config.rag.host,
        api_key=config.rag.api_key,
    )