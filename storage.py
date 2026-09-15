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
from datetime import datetime


class Storage:

    def __init__(self, result_dir: str | Path):
        self.result_dir = Path(result_dir)

        # 如果 results/ 不存在，就自動建立
        self.result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.experiment_path = self.result_dir / "experiment.json"

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
    def get_method(use_skill, use_rag):
        if use_skill and use_rag:
            return "skill_rag"
        elif use_skill:
            return "skill"
        elif use_rag:
            return "rag"
        else:
            return "basic"

    def create_experiment_record(self, config, dataset_size, skill=None):
        now = datetime.now()
        experiment_id = now.strftime("%Y%m%d_%H%M%S")
        started_at = now.isoformat(timespec="seconds")

        use_skill = config.experiment.use_skill
        use_rag = config.experiment.use_rag
        method = self.get_method(use_skill, use_rag)

        if skill is not None:
            skill_metadata = {
                "enabled": True,
                "name": skill.name,
                "source_type": skill.source_type,
                "source_path": str(skill.source_path),
            }

        else:
            skill_metadata = {
                "enabled": False,
                "name": None,
                "source_type": None,
                "source_path": None,
            }

        repair_model = "claude-haiku-4-5-20251001"

        metadata = {
            "experiment_id": experiment_id,
            "started_at": started_at,

            "method": method,

            "model": config.llm.model,
            "repair_model": repair_model,
            "skill": skill_metadata,
            "rag": {
                "enabled": use_rag,
            },
            "dataset": {
                "path": str(config.dataset.path),
                "total_questions": dataset_size,
            },

            "parameters": {
                "max_tokens": config.llm.max_tokens,
            },
        }

        with open(self.experiment_path,"w",encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return metadata
    
    def load_experiment_metadata(self):

        if not self.experiment_path.exists():
            return None
        with open(self.experiment_path,"r",encoding="utf-8",) as f:
            return json.load(f)
        
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