from os.path import exists, join

import pandas as pd

from src.main.data.oracles_dataset import get_oracles_dataset
from src.main.data.tokens_dataset import get_tokens_dataset
from src.main.util import root_dir


def generate_dataset(dataset_name: str, split: str, remove_empty_oracles: bool = False):
    """
    Generates the given dataset and saves the output as a ".pt" file.
    :param dataset_name: the dataset type
    :param split: the data split (i.e. "train" or "validation")
    :param remove_empty_oracles: whether to remove empty oracles from the data
    """
    # get all oracles
    print("Retrieving all oracles.")
    if dataset_name == "oracles":
        dataset = get_oracles_dataset(split=split)
    elif dataset_name == "tokens":
        dataset = get_tokens_dataset(split=split, use_retrieval=False)
    elif dataset_name == "tokens_retrieval":
        dataset = get_tokens_dataset(split=split, use_retrieval=True)
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")
    # remove empty oracles if necessary
    if remove_empty_oracles:
        print("Removing empty oracles.")
        dataset = dataset[dataset["label"] != "// No assertion"]
    print(f"Generated {len(dataset)} datapoints from the {dataset_name} {split} dataset.")
    # save final dataset
    artifact_name = join(root_dir, "dataset", f"{dataset_name}_{split}_dataset.json")
    dataset.to_json(artifact_name, orient='records')


def load_dataset(dataset_name: str, split: str) -> pd.DataFrame:
    """
    Loads the given dataset saved as a file on disk. Expects that the
    generate_dataset method has been run before.
    :param dataset_name: the dataset type
    :param split: the data split (i.e. "train" or "validation")
    :return: the dataset as a pandas dataframe
    """
    artifact_name = join(root_dir, "dataset", f"{dataset_name}_{split}_dataset.json")
    if not exists(artifact_name):
        raise ValueError("Unable to find dataset", artifact_name)
    return pd.read_json(artifact_name)
