from os.path import join

import pandas as pd

from src.main.data.oracles_dataset import get_oracles_dataset
from src.main.data.tokens_dataset import get_tokens_dataset
from src.main.util import root_dir

# all supported dataset names
dataset_names = ["oracles", "tokens", "tokens_retrieval"]


def _filter_empty_oracles(unfiltered_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all datapoints corresponding to an "empty" oracle.
    :param unfiltered_dataset: a dataset with empty and non-empty oracles
    :return: the dataset with empty oracles removed
    """
    return unfiltered_dataset


def generate_dataset(dataset_name: str, remove_empty_oracles: bool = False):
    """
    Generates the given dataset and saves the output as a ".pt" file.
    :param dataset_name: the dataset type
    :param remove_empty_oracles: whether to remove empty oracles from the data
    """
    # get all oracles
    if dataset_name == "oracles":
        dataset = get_oracles_dataset()
    elif dataset_name == "tokens":
        dataset = get_tokens_dataset(use_retrieval=False)
    elif dataset_name == "tokens_retrieval":
        dataset = get_tokens_dataset(use_retrieval=True)
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")
    print(dataset)
    # # remove empty oracles if necessary
    # if remove_empty_oracles:
    #     dataset = _filter_empty_oracles(dataset)
    # print(f"Generated {len(dataset)} datapoints from the {dataset_name} dataset.")
    # # save final dataset
    # artifact_name = join(root_dir, "dataset", f"{dataset_name}_dataset.json")
    # dataset.to_json(artifact_name, orient='records')


if __name__ == "__main__":
    generate_dataset("oracles")
