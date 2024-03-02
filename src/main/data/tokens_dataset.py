from os import walk
from os.path import join

import pandas as pd
from tqdm import tqdm

from src.main.util import root_dir


def _reformat_token_dp(raw_token_dp: pd.DataFrame) -> pd.DataFrame:
    pass


def _aggregate_grouped_token_dps(grouped_token_dps: pd.DataFrame, oracle_so_far: str) -> pd.DataFrame:
    grouped_token_dps = grouped_token_dps.reset_index()
    # collect all token information
    token_tuples = []
    for idx in grouped_token_dps.index:
        token_tuples.append((
            grouped_token_dps["token"][idx],
            grouped_token_dps["tokenClass"][idx],
            grouped_token_dps["tokenInfo"][idx]
        ))
    # aggregate remaining data
    agg_data = {
        "javadocTag": grouped_token_dps["javadocTag"][0],
        "methodJavadoc": grouped_token_dps["methodJavadoc"][0],
        "methodSourceCode": grouped_token_dps["methodSourceCode"][0],
        "oracleSoFar": oracle_so_far,
        "nextPossibleTokens": token_tuples
    }
    return pd.DataFrame(agg_data)


def _group_token_dps(raw_token_dps: pd.DataFrame) -> pd.DataFrame:
    grouped_token_dps = []
    for _, oracle_id_grouped_token_dps in raw_token_dps.groupby("oracleId"):
        for oracle_so_far,  oracle_so_far_grouped_token_dps in oracle_id_grouped_token_dps.groupby("oracleSoFar"):
            grouped_token_dps.append(_aggregate_grouped_token_dps(oracle_so_far_grouped_token_dps, str(oracle_so_far)))
    return pd.concat(grouped_token_dps)


def get_tokens_dataset() -> pd.DataFrame:
    """
    Gets all non-empty token datapoints from the tokens dataset, and
    re-formats each datapoint using the format specified in the top-level
    README.
    :return: the re-formatted token datapoints
    """
    dataset_dir = join(root_dir, "dataset", "tokens-dataset")
    token_dps_list = []
    for root, _, files in walk(dataset_dir):
        for file in tqdm(files):
            abs_path = join(root, file)
            raw_token_dps = pd.read_json(abs_path)
            if len(raw_token_dps) > 0:
                raw_token_dps = _group_token_dps(raw_token_dps)
                token_dps = raw_token_dps.apply(_reformat_token_dp, axis=1)
                token_dps_list.append(token_dps)
    return pd.concat(token_dps_list)


def main():
    get_tokens_dataset()


if __name__ == "__main__":
    main()
