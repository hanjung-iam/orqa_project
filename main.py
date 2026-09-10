from config import print_config
from cli import get_config_from_cli
from dataset import load_dataset
from skill import load_skill
from prompt import build_prompt, render_prompt


def main():
    config = get_config_from_cli()

    print_config(config)
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
        print("\n" + "-" * 30)
        print("Skill Instruction")
        print("-" * 30)
        print(skill.instruction)

    else:
        print("\nSkill is disabled.")

    question = dataset[0]
    prompt = build_prompt(
        question=question,
        skill=skill,
        retrieved_chunks=None,
    )
    print("[Prompt mode]")
    print(prompt.mode)
    print("[Rendered Prompt]")
    print(render_prompt(prompt))


if __name__ == "__main__":
    main()