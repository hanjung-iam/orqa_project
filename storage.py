"""
storage.py
負責 ORQA Project 的實驗結果保存與讀取。

results/
    ├── predictions.jsonl
    ├── invalid_answers.jsonl
    ├── evaluation.json
    └── wrong_predictions.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Storage:

    def __init__(self, result_dir: str | Path):

        # 將傳入的路徑轉成 Path，
        # 方便後續進行跨平台的檔案操作。
        self.result_dir = Path(result_dir)

        # 如果 results/ 不存在，就自動建立
        self.result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 每一題的完整 prediction record
        self.predictions_file = (
            self.result_dir / "predictions.jsonl"
        )

        # 判定為 invalid 的題目
        self.invalid_file = (
            self.result_dir / "invalid_answers.jsonl"
        )

        # evaluation summary
        self.evaluation_file = (
            self.result_dir / "evaluation.json"
        )

        # 所有答錯的題目
        self.wrong_predictions_file = (
            self.result_dir / "wrong_predictions.json"
        )

    @staticmethod
    def _append_jsonl(
        path: Path,
        record: dict[str, Any],
    ) -> None:

        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    @staticmethod
    def _write_json(path: Path,data: Any) -> None:
        
        with path.open("w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        records = []

        with path.open("r",encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        return records


    def save_prediction(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.predictions_file,record)

    def load_predictions(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.predictions_file)

    def get_completed_indices(self) -> set[int]:
        predictions = self.load_predictions()
        return {
            int(record["index"])
            for record in predictions
            if "index" in record
        }

    def save_invalid_answer(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.invalid_file, record)


    def load_invalid_answers(self ) -> list[dict[str, Any]]:
        return self._read_jsonl(self.invalid_file)


    def save_evaluation(self, result: dict[str, Any],) -> None:
        self._write_json(self.evaluation_file, result)


    def load_evaluation(self ) -> dict[str, Any] | None:
        if not self.evaluation_file.exists():
            return None

        with self.evaluation_file.open("r", encoding="utf-8",) as f:
            return json.load(f)


    def save_wrong_predictions(self, records: list[dict[str, Any]]) -> None:
        self._write_json(self.wrong_predictions_file, records)


    def load_wrong_predictions(self,) -> list[dict[str, Any]] | None:
        if not self.wrong_predictions_file.exists():
            return None

        with self.wrong_predictions_file.open("r",encoding="utf-8") as f:
            return json.load(f)


    def clear_predictions(self) -> None:
        """
        刪除目前的 prediction checkpoint
        storage.clear_predictions()
        """
        if self.predictions_file.exists():
            self.predictions_file.unlink()


    def print_status(self) -> None:
        """
        印出目前 Storage 的狀態
        用來快速確認目前已經完成多少題
        """
        predictions = self.load_predictions()
        invalid = self.load_invalid_answers()

        print("\n[Storage Status]")
        print(f"Predictions : {len(predictions)}")
        print(f"Invalid     : {len(invalid)}")
        print(f"Result dir  : {self.result_dir}")

     # 將 InferenceEngine 的 InferenceResult 轉換成可以保存的 dictionary 並寫入 predictions.jsonl
    def save_inference_result(self,question,result,) -> dict[str, Any]:
       
        llm_response = result.llm_response
        record = {
            "index": question.index,
            "question_type": (question.question_type),
            "prediction": result.prediction,
            "raw_response": result.raw_response,

            "repaired": result.repaired,
            "repair_response": (result.repair_response),

            "model": llm_response.model,
            "input_tokens": (llm_response.input_tokens),
            "output_tokens": (llm_response.output_tokens),

            "cache_creation_input_tokens": (llm_response.cache_creation_input_tokens),
            "cache_read_input_tokens": (llm_response.cache_read_input_tokens),
        }


        if result.repair_llm_response is not None:
            repair_response = (result.repair_llm_response)

            record.update({
                "repair_input_tokens": (repair_response.input_tokens),
                "repair_output_tokens": (repair_response.output_tokens),
                #"repair_cache_creation_input_tokens": (repair_response.cache_creation_input_tokens),
                #"repair_cache_read_input_tokens": (repair_response.cache_read_input_tokens),
            })

        else:
            record.update({
                "repair_model": None,
                "repair_input_tokens": 0,
                "repair_output_tokens": 0,
                #"repair_cache_creation_input_tokens": 0,
                #"repair_cache_read_input_tokens": 0,
            })

        self.save_prediction(record)

        return record


def create_storage(result_dir: str | Path) -> Storage:
    return Storage(result_dir)