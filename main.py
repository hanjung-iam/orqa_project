from config import create_config, print_config


def main():
    config = create_config(
        use_skill=False,
        use_rag=False,
    )

    print_config(config)


if __name__ == "__main__":
    main()