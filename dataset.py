import sys

from src.main.data import generate_dataset

# all supported dataset names
dataset_names = ["oracles", "tokens", "tokens_retrieval"]


def main():
    args = sys.argv[1:]
    assert len(args) >= 2, "Expected at least two arguments: [dataset_name], [split]"
    dataset_name = args[0]
    split = args[1]
    remove_empty_oracles = False if len(args) == 2 else bool(args[2])
    assert dataset_name in dataset_names
    assert split in ["train", "validation"]
    generate_dataset(dataset_name, split, remove_empty_oracles)


if __name__ == "__main__":
    main()
