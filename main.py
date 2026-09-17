from config import print_config
from cli import get_config_from_cli
from dataset import load_dataset
from skill import load_skill
from llm import create_llm_client
from inference import InferenceEngine
from parser import index_to_answer
from storage import create_storage
from evaluation import Evaluator
from tqdm import tqdm


def main():
    config = get_config_from_cli()
    #print_config(config)

    print("[Loading dataset]")

    dataset = load_dataset(config.dataset.path)
    print(f"datset path: {config.dataset.path}")
    print(f"Total questions: {len(dataset)}")

    skill = None

    if config.experiment.use_skill:
        print("[Loading Skill]")

        skill = load_skill(
            skill_path=config.skill.path,
            source_type=config.skill.source_type,
        )

        print(f"Skill name   : {skill.name}")
        print(f"Skill source : {skill.source_path}")
        print(f"Source type  : {skill.source_type}")

        print("\nSkill files:")
        for file_name in skill.files:
            print(f"  - {file_name}")


    print("[Creating client]")
    llm_client = create_llm_client(config)

    storage = create_storage(config.output.base_dir)
    print(f"Storage dir: {storage.result_dir}")

    storage.create_experiment_record(
        config=config,
        dataset_size=len(dataset),
        skill=skill,
    )

    print("[Creating inference engine]")
    engine = InferenceEngine(
        llm_client=llm_client,
        skill=skill,
    )
    print("[Running inference]")
    completed_indices = (storage.get_completed_indices())
    print(f"[Resume from question {len(completed_indices)}]")

    for question in tqdm(dataset,total=len(dataset),desc="Inference",unit="question"):

        if question.index in completed_indices:
            #skipped_count += 1
            print(
                f"[Skip Question {question.index}] "
                f"Already completed."
            )
            continue
        

        response = engine.run(question)

        if response.repaired:

            if response.repair_llm_response is not None:
                repair_text = response.repair_llm_response.text
            else:
                repair_text = None

            invalid_record = {
                "index": question.index,
                "question_type": question.question_type,

                "prediction_before_repair": None,
                "prediction_after_repair": response.prediction,

                "raw_response": response.raw_response,
                "repair_response": repair_text,
            }

            storage.save_invalid_answer(invalid_record)

        storage.save_inference_result(
            question=question,
            result=response,
        )

        completed = question.index+1

        if completed%100 == 0:
            print(f"Processed {completed} / {len(dataset)} questions")

        


    print("[Loading storage]")
    save_predictions = storage.load_predictions()
    print(f"Total saved predictions: {len(save_predictions)}")
    invalid_answers = storage.load_invalid_answers()
    print(f"Total saved invalid answers: {len(invalid_answers)}")

    print("\n[Running evaluation]")

    evaluator = Evaluator()
    evaluation_result = evaluator.evaluate(
        dataset=dataset,
        predictions=save_predictions,
    )
    evaluator.print_result(evaluation_result)
    storage.save_evaluation(evaluation_result.to_dict())
    storage.save_wrong_predictions(evaluation_result.wrong_predictions)


if __name__ == "__main__":
    main()