dataset_names = ["oracles", "tokens"]


def generate(dataset_name: str):
    """
    Generates the given dataset and saves the output as a ".pt" file.
    :param dataset_name: the dataset type
    """
    pass


def main():
    dataset_name = "oracles"
    assert dataset_name in dataset_names
    generate(dataset_name)


if __name__ == "__main__":
    main()
