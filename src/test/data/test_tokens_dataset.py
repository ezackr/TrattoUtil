import os

from src.main.data.tokens_dataset import get_tokens_dataset
from src.main.util.io import root_dir

example_tokens_dataset_path: str = os.path.join(root_dir, "src", "test", "resources", "tokens-dataset")


def test_get_oracles_dataset():
    dataset = get_tokens_dataset(example_tokens_dataset_path)
    actual_labels = list(dataset["label"])
    expected_labels = ["assertTrue(", "checkpoint", "!=", "// No assertion"]
    assert all([actual == expected for actual, expected in zip(actual_labels, expected_labels)])
