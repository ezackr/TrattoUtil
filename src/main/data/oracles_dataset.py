from os import walk
from os.path import join

import pandas as pd
from tqdm import tqdm

from src.main.util import root_dir


def _reformat_oracle_dp(raw_oracle_dp: pd.DataFrame) -> pd.DataFrame:
    """
    Re-formats an oracle datapoint into the new format, as described in the
    top-level README.
    :param raw_oracle_dp: the original oracle datapoint
    :return: the re-formatted oracle datapoint
    """
    output_col_name = "reformat_col"
    method_javadoc = raw_oracle_dp["methodJavadoc"] \
        .replace("    /**", "/**") \
        .replace("\n     *", "\n *") \
        .replace("\n   *", "\n *") \
        .replace("\n\t *", "\n *")
    method_signature = raw_oracle_dp["methodSourceCode"].split("{")[0]
    assertion_comment = f'// \"{raw_oracle_dp["javadocTag"]}\" assertion'.replace("\n", "\\n")
    assertion = f'assertTrue({raw_oracle_dp["oracle"].split(";")[0]});'
    raw_oracle_dp[output_col_name] = method_javadoc + "\n" + \
        method_signature + " {\n}\n\n" + \
        assertion_comment + "\n" + \
        assertion
    return raw_oracle_dp[output_col_name]


def _read_nonempty_oracle_dps(abs_path: str) -> pd.DataFrame:
    """
    Reads all non-empty oracle datapoints from a JSON file as a pandas
    dataframe.
    :param abs_path: the path to the JSON file of oracle datapoints
    :return: the corresponding pandas dataframe
    """
    raw_oracle_dps = pd.read_json(abs_path)
    if len(raw_oracle_dps) == 0:
        return raw_oracle_dps
    return raw_oracle_dps[(raw_oracle_dps["oracle"] != ";") & (raw_oracle_dps["methodJavadoc"] != "")]


def get_oracles_dataset() -> pd.DataFrame:
    """
    Gets all non-empty oracles from the oracles dataset and re-formats each
    datapoint using the format specified in the top-level README.
    :return: the re-formatted oracles
    """
    dataset_dir = join(root_dir, "dataset", "oracles-dataset")
    oracle_dps_list = []
    for root, _, files in walk(dataset_dir):
        for file in tqdm(files):
            abs_path = join(root, file)
            raw_oracle_dps = _read_nonempty_oracle_dps(abs_path)
            if len(raw_oracle_dps) > 0:
                oracle_dps = raw_oracle_dps.apply(_reformat_oracle_dp, axis=1)
                oracle_dps_list.append(oracle_dps)
    all_oracle_dps = pd.concat(oracle_dps_list).reset_index()
    all_oracle_dps.rename(columns={0: "text"}, inplace=True)
    all_oracle_dps.drop(columns=["index"], inplace=True)
    return all_oracle_dps
