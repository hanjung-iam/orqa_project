from config import print_config
from cli import get_config_from_cli
from dataset import load_dataset


def main():
    config = get_config_from_cli()

    print_config(config)
    print("[Loading dataset]")

    dataset = load_dataset(config.dataset.path)
    print(f"datset path: {config.dataset.path}")
    print(f"Total questions: {len(dataset)}")

    question = dataset[0]
    print(f"index: {question.index}")
    print(f"question_type: {question.question_type}")
    print(f"context: {question.context}")
    print(f"question: {question.question}")
    print(f"options: {question.options}")
    print(f"target answer:{question.target_answer}")

if __name__ == "__main__":
    main()