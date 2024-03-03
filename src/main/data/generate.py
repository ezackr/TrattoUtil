from os.path import join

from src.main.data.oracles_dataset import get_oracles_dataset
from src.main.data.tokens_dataset import get_tokens_dataset
from src.main.util import root_dir

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
    print(f"Generated {len(dataset)} datapoints from the {dataset_name} dataset.")
    artifact_name = join(root_dir, "dataset", f"{dataset_name}_dataset.json")
    dataset.to_json(artifact_name, orient='records')
