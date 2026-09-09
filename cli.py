"""
cli.py
負責專案的 Command Line Interface
"""

from pathlib import Path

from config import ExperimentConfig
from config import create_config

def ask_choices(message: str) -> bool:
    while True:
        answer = input(
            f"{message} [y/n]: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print(
            "Invalid input. Please enter 'y' or 'n'."
        )

def ask_skill_path() -> Path:

    while True:

        path_string = input(
            "Please enter the path to your Skill file: "
        ).strip()

        if not path_string:
            print("Skill path cannot be empty.")
            continue

        path = Path(path_string)
        if not path.exists():

            print(f"Skill file does not exist: {path}")
            print("Please enter a valid Skill file path.")

            continue

        if not path.is_file():

            print(f"The specified Skill path is not a file: {path}")

            continue

        return path

def collect_user_requirements() -> dict:
    print()
    print("=" * 30)
    print("ORQA Experiment Configuration")
    print("=" * 30)

    # skill use?
    use_skill = ask_choices( "Do you want to use Skill?")

    skill_path = None
    if use_skill:
        skill_path = ask_skill_path()

    # rag use?
    use_rag = ask_choices("Do you want to use RAG?")

    requirements = {
        "use_skill": use_skill,
        "skill_path": skill_path,
        "use_rag": use_rag,
    }

    print()
    print("-" * 30)
    print("Your experiment settings:")
    print("-" * 30)

    print(f"Use Skill : {requirements['use_skill']}")
    print( f"Skill Path: {requirements['skill_path']}")
    print(f"Use RAG   : {requirements['use_rag']}")

    print("-" * 30)

    return requirements


def get_config_from_cli() -> ExperimentConfig:

    requirements = collect_user_requirements()

    config = create_config(
        use_skill=requirements["use_skill"],
        use_rag=requirements["use_rag"],
        skill_path=requirements["skill_path"],
    )
    return config
