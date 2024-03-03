from src.main.data.oracles_dataset import get_oracles_dataset
from src.main.data.tokens_dataset import get_tokens_dataset

# all supported dataset names
dataset_names = ["oracles", "tokens"]


def generate_dataset(dataset_name: str):
    """
    Generates the given dataset and saves the output as a ".pt" file.
    :param dataset_name: the dataset type
    """
    if dataset_name == "oracles":
        dataset = get_oracles_dataset()
    elif dataset_name == "tokens":
        dataset = get_tokens_dataset()
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")
    print(len(dataset))


def main():
    dataset_name = "oracles"
    generate_dataset(dataset_name)


if __name__ == "__main__":
    main()
