"""
evaluation.py
負責 ORQA prediction 的評估
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any

from parser import index_to_answer


@dataclass
class EvaluationResult:

    total: int
    correct: int
    wrong: int
    invalid: int
    accuracy: float
    question_type_stats: dict[str, dict[str, Any]]
    wrong_predictions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "wrong": self.wrong,
            "invalid": self.invalid,
            "accuracy": self.accuracy,
            "question_type_stats": self.question_type_stats,
            "wrong_predictions": self.wrong_predictions,
        }

class Evaluator:

    def evaluate(
        self,
        dataset,
        predictions: list[dict[str, Any]],
    ) -> EvaluationResult:

        prediction_map = {
            int(record["index"]): record
            for record in predictions
            if "index" in record
        }

        total = 0
        correct = 0
        wrong = 0
        invalid = 0

        # 保存所有答錯的題目
        wrong_predictions = []

        question_type_stats = defaultdict(
            lambda: {
                "total": 0,
                "correct": 0,
                "wrong": 0,
                "invalid": 0,
            }
        )

        for question in dataset:

            index = int(question.index)

            total += 1
            question_type = question.question_type

            stats = question_type_stats[question_type]

            stats["total"] += 1

            record = prediction_map.get(index)

            # 如果 prediction 完全不存在 視為 invalid / missing
            if record is None:
                prediction = -1
                raw_response = ""
            else:

                prediction = int(record.get("prediction",-1))
                raw_response = record.get("raw_response", "")

            if prediction == -1:

                invalid += 1
                wrong += 1

                stats["invalid"] += 1
                stats["wrong"] += 1

                wrong_predictions.append(
                    self._build_wrong_record(
                        question=question,
                        prediction=prediction,
                        raw_response=raw_response,
                    )
                )
                continue

            if prediction == question.target_answer:

                correct += 1
                stats["correct"] += 1

            else:

                wrong += 1
                stats["wrong"] += 1

                wrong_predictions.append(
                    self._build_wrong_record(
                        question=question,
                        prediction=prediction,
                        raw_response=raw_response,
                    )
                )

        if total == 0:
            accuracy = 0.0

        else:
            accuracy = correct / total


        question_type_result = {}

        for question_type, stats in question_type_stats.items():

            type_total = stats["total"]

            if type_total == 0:
                type_accuracy = 0.0
            else:
                type_accuracy = (stats["correct"] / type_total)

            question_type_result[question_type] = {
                "total": type_total,
                "correct": stats["correct"],
                "wrong": stats["wrong"],
                "invalid": stats["invalid"],
                "accuracy": type_accuracy,
            }


        question_type_result = dict(
            sorted(
                question_type_result.items(),
                key=self._question_type_sort_key,
            )
        )

        return EvaluationResult(
            total=total,
            correct=correct,
            wrong=wrong,
            invalid=invalid,
            accuracy=accuracy,
            question_type_stats=question_type_result,
            wrong_predictions=wrong_predictions,
        )


    @staticmethod
    def _build_wrong_record(
        question,
        prediction: int,
        raw_response: str,
    ) -> dict[str, Any]:

        return {
            "index": question.index,
            "question_type": (question.question_type),
            "prediction": prediction,
            "prediction_letter": (
                index_to_answer(prediction)
                if prediction != -1
                else None
            ),
            "ground_truth": (question.target_answer),
            "ground_truth_letter": (index_to_answer(question.target_answer)),
            "question": question.question,
            "context": question.context,
            "options": question.options,
            "raw_response": raw_response,
        }

    @staticmethod
    def _question_type_sort_key(item,) -> int:

        question_type = item[0]

        if question_type.startswith("Q"):
            try:
                return int(question_type[1:])
            except ValueError:
                pass
        return 999999


    @staticmethod
    def print_result(result: EvaluationResult,) -> None:

        print("\n" + "=" * 30)
        print("ORQA Evaluation Result")
        print("=" * 30)

        print(f"Total   : {result.total}")
        print(f"Correct : {result.correct}")
        print(f"Wrong   : {result.wrong}")
        print(f"Invalid : {result.invalid}")
        print(f"Accuracy: {result.accuracy:.4f}")

        print("\n" + "-" * 30)
        print("Question Type Statistics")
        print("-" * 30)

        print(
            f"{'Type':<8}"
            f"{'Total':<10}"
            f"{'Correct':<10}"
            f"{'Wrong':<10}"
            f"{'Invalid':<10}"
            f"{'Accuracy':<10}"
        )

        for question_type, stats in (result.question_type_stats.items()):
            print(
                f"{question_type:<8}"
                f"{stats['total']:<10}"
                f"{stats['correct']:<10}"
                f"{stats['wrong']:<10}"
                f"{stats['invalid']:<10}"
                f"{stats['accuracy']:.4f}"
            )

        print("\n" + "-" * 30)
        print(f"Wrong Predictions : {len(result.wrong_predictions)}")
        print("-" * 30)

def evaluate_predictions(dataset,predictions: list[dict[str, Any]]) -> EvaluationResult:
    evaluator = Evaluator()
    return evaluator.evaluate(
        dataset=dataset,
        predictions=predictions,
    )