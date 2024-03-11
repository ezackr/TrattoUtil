from os import walk
from os.path import join
import random
import re
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from src.main.util import root_dir


def _get_method_retrieval_information(method_source_code: str) -> str:
    """
    Gets the method Javadoc and signature of a token. If the method does not
    have a Javadoc, then only the signature is used. The returned string will
    include annotations.
    :param method_source_code: the source code of a method
    :return: the method Javadoc and signature
    """
    if "{" not in method_source_code:
        # "source code" only includes method signature
        return method_source_code + " {\n}\n\n"
    # separate javadoc comment and source code
    comment_indexes = [i for i in range(len(method_source_code) - 1) if method_source_code[i:i+2] == "*/"]
    if comment_indexes:
        method_javadoc = method_source_code[:comment_indexes[-1] + 2]
        method_signature = method_source_code[comment_indexes[-1] + 2:].split("{")[0]
        return method_javadoc + method_signature + " {\n}\n\n"
    else:
        return method_source_code.split("{")[0] + " {\n}\n\n"


def _get_retrieval_information(next_possible_tokens: pd.DataFrame) -> str:
    """
    Gets all information for the next possible tokens
    :param next_possible_tokens: all next possible tokens
    :return: a string with information for each possible token
    """
    retrieval_info = ""
    for token_info in next_possible_tokens:
        if token_info[1] == "MethodName":
            method_source_code = token_info[2][2]
            if method_source_code:
                retrieval_info += _get_method_retrieval_information(method_source_code)
    return retrieval_info


def _get_next_token(next_possible_tokens: Tuple[str, str, str, str]) -> str:
    """
    Gets the next token in an oracle from a list of possible tokens
    :param next_possible_tokens: all next possible tokens
    :return: the next token in the oracleSoFar
    """
    for token_info in next_possible_tokens:
        if token_info[-1]:
            return token_info[0]
    raise ValueError("Unable to identify next possible token from list:", next_possible_tokens)


def _reformat_token_dp(grouped_token_dp: pd.DataFrame, use_retrieval: bool) -> pd.DataFrame:
    """
    Re-formats a token datapoint into the new format, as described in the
    top-level README.
    :param grouped_token_dp: token datapoints grouped by "oracleId" and
    "oracleSoFar"
    :return: the re-formatted token datapoint
    """
    # get retrieval information
    if use_retrieval:
        retrieval_info = _get_retrieval_information(grouped_token_dp["nextPossibleTokens"])
    else:
        retrieval_info = ""
    # get prompt and label
    method_javadoc = grouped_token_dp["methodJavadoc"].replace("    /**", "/**")
    method_javadoc = re.sub(r"\n[ \t]*\*", "\n *", method_javadoc, flags=re.MULTILINE)
    method_signature = grouped_token_dp["methodSourceCode"].split("{")[0]
    assertion_comment = f'// \"{grouped_token_dp["javadocTag"]}\" assertion'
    assertion_comment = re.sub(r"\n\s*", " ", assertion_comment)
    next_token = _get_next_token(grouped_token_dp["nextPossibleTokens"])
    token_values = [");" if t_info[0] == ";" else t_info[0] for t_info in grouped_token_dp["nextPossibleTokens"]]
    if next_token == ";" and grouped_token_dp["oracleSoFar"] == "":
        assertion_so_far = ""
        token_values = ["assertTrue", "// No assertion"]
        label = "// No assertion"
    else:
        assertion_so_far = f'assertTrue({grouped_token_dp["oracleSoFar"]}'
        label = next_token if next_token != ";" else ");"
    random.shuffle(token_values)
    next_possible_tokens_comment = f"// Next possible tokens: {token_values}"
    grouped_token_dp["prompt"] = retrieval_info + method_javadoc + "\n" + method_signature + " {\n}\n\n" + \
        assertion_comment + "\n" + next_possible_tokens_comment + "\n" + assertion_so_far
    grouped_token_dp["label"] = label
    return grouped_token_dp[["prompt", "label"]]


def _aggregate_grouped_token_dps(grouped_token_dps: pd.DataFrame, oracle_so_far: str) -> pd.DataFrame:
    """
    Aggregates a group of original token datapoints into a single datapoint.
    :param grouped_token_dps: the grouped raw token datapoints, each from the
    same original oracleId and oracleSoFar features
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
            grouped_token_dps["tokenInfo"][idx],
            grouped_token_dps["label"][idx]
        ))
    # aggregate datapoints
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


def _read_raw_token_dps(abs_path: str) -> pd.DataFrame:
    """
    Reads all token datapoints from a JSON file as a pandas dataframe. This
    method filters datapoints if their "methodJavadoc" is empty.
    :param abs_path: the path to the JSON file of token datapoints
    :return: the corresponding pandas dataframe
    """
    raw_token_dps = pd.read_json(abs_path)
    if len(raw_token_dps) == 0:
        return raw_token_dps
    return raw_token_dps[raw_token_dps["methodJavadoc"] != ""]


def _read_raw_token_dps_dir(dataset_dir: str) -> pd.DataFrame:
    """
    Reads all token datapoints from a directory of JSON files as a pandas
    dataframe. This method filters datapoints if their "methodJavadoc" is
    empty.
    :param dataset_dir: a directory of JSON files of token datapoints
    :return: the corresponding pandas dataframe
    """
    raw_token_dps_list = []
    for root, _, files in walk(dataset_dir):
        for file in tqdm(files):
            abs_path = join(root, file)
            raw_token_dps = _read_raw_token_dps(abs_path)
            if len(raw_token_dps) > 0:
                raw_token_dps_list.append(raw_token_dps)
    return pd.concat(raw_token_dps_list).reset_index()


def _create_starting_token_dp(token_dp: pd.Series) -> pd.Series:
    """
    Creates the starting token datapoint for a corresponding token datapoint.
    This token datapoint starts with an empty string, and determines whether
    he model generate an oracle.
    :param token_dp: a token datapoint
    :return: the corresponding starting token datapoint
    """
    lines = token_dp["prompt"].split("\n")
    lines[-1] = ""
    token_values = ["assertTrue", "// No assertion"]
    random.shuffle(token_values)
    lines[-2] = f"// Next possible tokens: {token_values}"
    token_dp["prompt"] = "\n".join(lines)
    token_dp["label"] = "assertTrue("
    return token_dp


def _add_starting_token_dps(reformatted_token_dps: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all starting token datapoints for all token datapoints in the
    Tokens Dataset.
    :param reformatted_token_dps: all reformatted token datapoints
    :return: all reformatted token datapoints, with the added starting token
    datapoints
    """
    augmented_token_dps = []
    for _, token_dp in reformatted_token_dps.iterrows():
        if token_dp["prompt"].endswith("assertTrue("):
            augmented_token_dps.append(_create_starting_token_dp(token_dp.copy()))
        augmented_token_dps.append(token_dp)
    return pd.DataFrame(augmented_token_dps).reset_index()


def get_tokens_dataset(dataset_dir: str = None, use_retrieval: bool = False) -> pd.DataFrame:
    """
    Gets all non-empty token datapoints from the tokens dataset, and
    re-formats each datapoint using the format specified in the top-level
    README.
    :return: the re-formatted token datapoints
    """
    if not dataset_dir:
        dataset_dir_name = "tokens-retrieval-dataset" if use_retrieval else "tokens-dataset"
        dataset_dir = join(root_dir, "dataset", dataset_dir_name)
    all_raw_token_dps = _read_raw_token_dps_dir(dataset_dir)
    all_grouped_token_dps = _group_token_dps(raw_token_dps=all_raw_token_dps)
    all_token_dps = all_grouped_token_dps.apply(lambda x: _reformat_token_dp(x, use_retrieval), axis=1)
    all_token_dps = _add_starting_token_dps(all_token_dps)
    all_token_dps.drop(columns=["index"], inplace=True)
    return all_token_dps
