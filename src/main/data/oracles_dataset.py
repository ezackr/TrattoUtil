import itertools
from os import walk
from os.path import exists
from os.path import join
import re

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
    # get prompt
    method_javadoc = raw_oracle_dp["methodJavadoc"].replace("    /**", "/**")
    method_javadoc = re.sub(r"\n[ \t]*\*", "\n *", method_javadoc, flags=re.MULTILINE)
    method_signature = raw_oracle_dp["methodSourceCode"].split("{")[0]
    assertion_comment = f'// \"{raw_oracle_dp["javadocTag"]}\" assertion'
    assertion_comment = re.sub(r"\n\s*", " ", assertion_comment)
    raw_oracle_dp["prompt"] = method_javadoc + "\n" + method_signature + " {\n}\n\n" + assertion_comment + "\n"
    # get label
    if raw_oracle_dp["oracle"] == ";":
        raw_oracle_dp["label"] = "// No assertion"
    else:
        raw_oracle_dp["label"] = f'assertTrue({raw_oracle_dp["oracle"].split(";")[0]});'
    return raw_oracle_dp[["prompt", "label"]]


def _read_nonempty_oracle_dps(abs_path: str) -> pd.DataFrame:
    """
    Reads all oracle datapoints from a JSON file as a pandas dataframe. This
    method filters datapoints if their "methodJavadoc" is empty.
    :param abs_path: the path to the JSON file of oracle datapoints
    :return: the corresponding pandas dataframe
    """
    raw_oracle_dps = pd.read_json(abs_path)
    if len(raw_oracle_dps) == 0:
        return raw_oracle_dps
    return raw_oracle_dps[raw_oracle_dps["methodJavadoc"] != ""]


def get_oracles_dataset(dataset_dir: str = None) -> pd.DataFrame:
    """
    Gets all non-empty oracles from the oracles dataset and re-formats each
    datapoint using the format specified in the top-level README.
    :return: the re-formatted oracles
    """
    if not dataset_dir:
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
    all_oracle_dps.drop(columns=["index"], inplace=True)
    return all_oracle_dps


def tokenize_prompt(tokenizer, oracle_dp: pd.DataFrame) -> pd.DataFrame:
    """
    Tokenizes an oracle datapoint using a given tokenizer. Generates
    input_ids, labels, and an attention mask.
    :param tokenizer: a tokenizer
    :param oracle_dp: a token datapoint
    :return: the corresponding input ids, labels, and attention mask
    """
    prompt_tok = [tokenizer.encode(f"{tokenizer.bos_token} {(oracle_dp['prompt']).strip()}", add_special_tokens=False)]
    answer_tok = [tokenizer.encode(f"{oracle_dp['label'].strip()} {tokenizer.eos_token}", add_special_tokens=False)]
    input_tok = list(itertools.chain.from_iterable(zip(prompt_tok, answer_tok)))
    input_tok = list(itertools.chain(*(t for t in input_tok)))
    oracle_dp["input_ids"] = input_tok
    oracle_dp["labels"] = input_tok.copy()
    oracle_dp["attention_mask"] = [1] * len(input_tok)
    return oracle_dp[["input_ids", "labels", "attention_mask"]]


def get_custom_dataset(dataset_config, tokenizer, split: str):
    """
    The method used by the llama-recipes repository for loading the oracles
    dataset for CodeLLaMa fine-tuning. The parameters of this method are
    hard-coded for compatibility with the llama-recipes framework, such that
    some parameters may be unused.
    :param dataset_config: dataset configuration
    :param tokenizer: input tokenizer
    :param split: the train or test data split
    :return: a HuggingFace dataset
    """
    artifact_path = join(root_dir, "dataset", "oracles_dataset.json")
    if not exists(artifact_path):
        raise ValueError("Unable to locate dataset file:", artifact_path)
    dataset = pd.read_json(artifact_path)
    dataset = dataset.apply(lambda x: tokenize_prompt(tokenizer, x), axis=1)
