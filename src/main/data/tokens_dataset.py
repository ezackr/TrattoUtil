from os import walk
from os.path import join

import pandas as pd
from tqdm import tqdm

from src.main.util import root_dir


def _get_retrieval_information(next_possible_tokens: pd.DataFrame) -> str:
    pass


def _reformat_token_dp(grouped_token_dp: pd.DataFrame, use_retrieval: bool) -> pd.DataFrame:
    """
    Re-formats a token datapoint into the new format, as described in the
    top-level README.
    :param grouped_token_dp:
    :return:
    """
    output_col_name = "reformat_col"
    if use_retrieval:
        retrieval_info = _get_retrieval_information(grouped_token_dp["nextPossibleTokens"]) + "\n\n"
    else:
        retrieval_info = ""
    method_javadoc = grouped_token_dp["methodJavadoc"].replace("    /**", "/**").replace("\n     *", "\n *")
    method_signature = grouped_token_dp["methodSourceCode"].split("{")[0]
    assertion_comment = f'// \"{grouped_token_dp["javadocTag"]}\" assertion'.replace("\n", "\\n")
    token_values = [token_info[0] for token_info in grouped_token_dp["nextPossibleTokens"]]
    next_possible_tokens_comment = f"// Next possible tokens: {token_values}"
    assertion = f'assertTrue({grouped_token_dp["oracleSoFar"]}'
    grouped_token_dp[output_col_name] = retrieval_info + \
        method_javadoc + "\n" + \
        method_signature + " {\n}\n\n" + \
        assertion_comment + "\n" + \
        next_possible_tokens_comment + "\n" + \
        assertion
    return grouped_token_dp[output_col_name]


def _aggregate_grouped_token_dps(grouped_token_dps: pd.DataFrame, oracle_so_far: str) -> pd.DataFrame:
    """
    Aggregates a group of original token datapoints into a single token
    datapoint.
    :param grouped_token_dps: the original token datapoints, each sharing the
    same oracleId and oracleSoFar
    :param oracle_so_far: the shared oracleSoFar feature
    :return: an aggregated token datapoint
    """
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
        "nextPossibleTokens": [token_tuples]
    }
    return pd.DataFrame(agg_data)


def _group_token_dps(raw_token_dps: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the original token datapoints into groups based on their
    corresponding oracleId and oracleSoFar features.
    :param raw_token_dps: the original token datapoints
    :return: the aggregated token datapoints
    """
    grouped_token_dps = []
    for _, oracle_id_grouped_token_dps in raw_token_dps.groupby("oracleId"):
        for oracle_so_far,  oracle_so_far_grouped_token_dps in oracle_id_grouped_token_dps.groupby("oracleSoFar"):
            grouped_token_dps.append(_aggregate_grouped_token_dps(oracle_so_far_grouped_token_dps, str(oracle_so_far)))
    return pd.concat(grouped_token_dps)


def get_tokens_dataset(use_retrieval: bool = False) -> pd.DataFrame:
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
                grouped_token_dps = _group_token_dps(raw_token_dps)
                token_dps = grouped_token_dps.apply(lambda x: _reformat_token_dp(x, use_retrieval), axis=1)
                token_dps_list.append(token_dps)
    all_token_dps = pd.concat(token_dps_list).reset_index()
    all_token_dps.rename(columns={0: "text"}, inplace=True)
    all_token_dps.drop(columns=["index"], inplace=True)
    return all_token_dps
