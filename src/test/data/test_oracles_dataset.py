import os

from src.main.data.oracles_dataset import get_oracles_dataset
from src.main.util.io import root_dir

example_oracles_dataset_path: str = os.path.join(root_dir, "src", "test", "resources", "oracles-dataset")


def test_get_oracles_dataset():
    dataset = get_oracles_dataset(example_oracles_dataset_path)
    prompt = dataset["prompt"][0]
    label = dataset["label"][0]
    assert prompt == """/**
 * Returns the current count.
 * @param checkpoint must not be null
 * @return the current count
 */
public int getCount(MyObject checkpoint)  {
}

// "@param checkpoint must not be null" assertion
"""
    assert label == "assertTrue(checkpoint != null);"
