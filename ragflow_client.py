from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from ragflow_sdk import RAGFlow
import time

@dataclass(frozen=True)
class RAGFlowDatasetInfo:
    id: str # RAGFlow Dataset ID
    name: str # Dataset name
    embedding_model: Optional[str]
    chunk_method: Optional[str]
    document_count: Optional[int]
    chunk_count: Optional[int]

@dataclass(frozen=True)
class RAGFlowDocInfo:
    id: str # 文件的ID
    name: str
    dataset_id: str
    chunk_count: Optional[int] 
    run: Optional[str] # 文件處理狀態

class RAGFlowClient:

    def __init__(self,host: str,api_key: str):
        if not host:
            raise ValueError("RAGFlow host cannot be empty.")
        if not api_key:
            raise ValueError("RAGFlow API key cannot be empty.")

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
            raise ValueError("Dataset name cannot be empty.")

        if not embedding_model:
            raise ValueError("Embedding model cannot be empty.")

        if not chunk_method:
            raise ValueError("Chunk method cannot be empty." )

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
        
        pdf_dataset, _ = (
            self.get_or_create_dataset(
                name=pdf_dataset_name,
                embedding_model=embedding_model,
                chunk_method=chunk_method,
                description=("ORQA knowledge base from PDF files."),
            )
        )

        md_dataset, _ = (
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

    def upload_file(
        self,
        dataset_name: str,
        file_path: str | Path,
    ) -> RAGFlowDocInfo:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        dataset = self.get_dataset_by_name(dataset_name)
        if dataset is None:
            raise RuntimeError(f"Dataset not found: {dataset_name}")

        with path.open("rb") as file:
            file_bytes = file.read()

        uploaded_doc = (dataset.upload_documents([
            {
                "display_name": path.name,
                "blob": file_bytes,
            }
        ]))
        if not uploaded_doc:
            raise RuntimeError(f"returned no uploaded doc for file: {path}")

        document = uploaded_doc[0]
        return self._to_doc_info(document)

    # 取得 dataset 裡的所有文件
    def get_documents(self, dataset_name: str):
        dataset = self.get_dataset_by_name(dataset_name)
        if dataset is None:
            raise RuntimeError(f"Dataset {dataset_name} does not exist.")
        return dataset.list_documents(page=1, page_size=100)

    # 根據 dataset name & document name 找文件
    def get_document(self, dataset_name: str, document_name: str):
        documents = self.get_documents(dataset_name)

        for doc in documents:
            if doc.name == document_name: return doc
        return None

    # upload multiple files
    def upload_files(
        self,
        dataset_name: str,
        file_paths: list[str | Path],
    ) -> list[RAGFlowDocInfo]:
        dataset = self.get_dataset_by_name(dataset_name)
        if dataset is None:
            raise RuntimeError(f"RAGFlow dataset not found: {dataset_name}")
        doc_list = []
        for file_path in file_paths:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")
            
            with path.open("rb") as file:
                file_bytes = file.read()
            doc_list.append({
                "display_name": path.name,
                "blob": file_bytes,
            })

        uploaded_doc = (dataset.upload_documents(doc_list))
        return[
            self._to_doc_infos(doc) for doc in uploaded_doc
        ]

    def parse_documents(
        self,
        dataset_name: str,
        document_ids: list[str],
    ) -> list[RAGFlowDocInfo]:
        if not document_ids:
            raise ValueError("document_ids cannot be empty.")
        dataset = self.get_dataset_by_name(dataset_name)
        if dataset is None:
            raise RuntimeError(f"dataset {dataset_name} does not exist.")

        finished_doc = dataset.parse_documents(document_ids)
        result = []

        for doc in finished_doc:
            id = doc[0]
            run = doc[1]
            chunk_count = doc[2]

            docs = dataset.list_documents(id=id)
            if not docs:
                raise RuntimeError(f"document {id} cannot be found fter parsing")
            document = docs[0]
            result.append(self._to_doc_info(document))

        return result

    def async_parse_doc(
        self,
        dataset_name: str,
        document_ids: list[str]
    )-> None:
        dataset = self.get_dataset_by_name(dataset_name)
        dataset.async_parse_documents(document_ids)

    def get_document_status(
        self,
        dataset_name: str,
        document_id: str
    ) -> RAGFlowDocInfo:
        dataset = self.get_dataset_by_name(dataset_name)
        if dataset is None:
            raise RuntimeError(f"Dataset {dataset_name} does not exists.")
        documents = dataset.list_documents(id=document_id) 
        if not documents:
            raise RuntimeError(f"Document {document_id} does not exist.")
        return self._to_doc_info(documents[0])

    # 等待指定文件 parsing 完成
    def wait_for_parsing(
        self,
        dataset_name: str,
        document_ids: list[str],
        poll_interval: int = 2, # 每隔2秒向RAGFlow查詢一次
        timeout: int = 3600 # 最常等待時間(秒)
    ) -> list[RAGFlowDocInfo]:

        if not document_ids:
            raise ValueError("document_ids connot be empty")
        start_time = time.time()
        pending = set(document_ids)

        latest_status = {}

        print("[Waiting for RAGFlow Document parsing]")
        while pending:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError("RAGFlow document parsing timed out")

            for id in list(pending):
                status = self.get_document_status(
                    dataset_name=dataset_name,
                    document_id=id
                )

                latest_status[id] = status

                print(f"[RAGFlow] {status.name} | {status.run}")

                run = status.run.upper() if isinstance(status.run, str) else ""
                if run == "DONE":
                    pending.discard(id)
                elif run == "FAIL":
                    raise RuntimeError(f"RAGFlow parsing failed for doc {status.name}")
                elif run == "CANCEL":
                    raise RuntimeError(f"RAGFlow parsing was cancilled for doc {status.name}")

            if pending:
                time.sleep(poll_interval)

        print("All RAGFlow docs finished parsing")
        return [
            latest_status[document_id]
            for document_id in document_ids
        ]
            

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

    @staticmethod
    def _to_doc_info(document) -> RAGFlowDocInfo:
        return RAGFlowDocInfo(
            id=getattr(document, "id", ""),
            name=getattr(document, "name", ""),
            dataset_id=getattr(document, "dataset_id", ""),
            chunk_count=getattr(document,"chunk_count",None),
            run=getattr(document,"run",None),
        )


def create_ragflow_client(config) -> RAGFlowClient:

    return RAGFlowClient(
        host=config.rag.host,
        api_key=config.rag.api_key,
    )