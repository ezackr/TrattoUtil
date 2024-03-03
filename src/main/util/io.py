import os

import pandas as pd

PANDAS_SEPARATOR: str = "[TRATTO-SEPARATOR]"


def get_parent_dir(path: str, level: int = 1) -> str:
    """
    Returns the (level)-th parent directory of a given path
    :param path: the root path
    :param level: the number of parent directories above the root path
    :return: the (level)-th parent directory of the given path
    :raise ValueError: if the given level is less than 0
    """
    if level == 0:
        return path
    if level < 0:
        raise ValueError(f"Parent directory level must be greater than 0. Actual: {level}")
    for _ in range(level):
        path = os.path.dirname(path)
    return path


def save_dataframe(dataset: pd.DataFrame):
    """
    Saves a pandas dataframe using a special separator to avoid accidentally
    overwriting the original data
    :param dataset: a pandas dataframe
    """
    dataset.to_csv(sep=PANDAS_SEPARATOR)


def load_dataframe(artifact_name: str) -> pd.DataFrame:
    """
    Loads a CSV file of a pandas dataframe that has been saved using
    ``src.main.util.io.save_dataframe``. The aforementioned method uses a
    special separator to avoid overwriting data.
    :param artifact_name: a CSV file
    :return: the corresponding pandas dataframe
    """
    return pd.read_csv(artifact_name, sep=PANDAS_SEPARATOR)


current_path: str = os.path.abspath(__file__)
root_dir: str = get_parent_dir(current_path, level=4)
