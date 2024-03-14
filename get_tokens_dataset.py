import itertools

from datasets import Dataset
import pandas as pd

from src.main.data import load_dataset


def tokenize_prompt(tokenizer, token_dp: pd.DataFrame) -> pd.DataFrame:
    """
    Tokenizes a token datapoint using a given tokenizer. Generates
    input_ids, labels, and an attention mask.
    :param tokenizer: a tokenizer
    :param token_dp: a token datapoint
    :return: the corresponding input ids, labels, and attention mask
    """
    prompt_tok = [tokenizer.encode(f"{tokenizer.bos_token} {(token_dp['prompt']).strip()}", add_special_tokens=False)]
    answer_tok = [tokenizer.encode(f"{token_dp['label'].strip()} {tokenizer.eos_token}", add_special_tokens=False)]
    input_tok = list(itertools.chain.from_iterable(zip(prompt_tok, answer_tok)))
    input_tok = list(itertools.chain(*(t for t in input_tok)))
    token_dp["input_ids"] = input_tok
    token_dp["labels"] = input_tok.copy()
    token_dp["attention_mask"] = [1] * len(input_tok)
    return token_dp[["input_ids", "labels", "attention_mask"]]


def get_custom_dataset(dataset_config, tokenizer, split: str):
    """
    The method used by the llama-recipes repository for loading the tokens
    dataset for CodeLLaMa fine-tuning. The parameters of this method are
    hard-coded for compatibility with the llama-recipes framework, such that
    some parameters may be unused.
    :param dataset_config: dataset configuration
    :param tokenizer: input tokenizer
    :param split: the train or test data split
    :return: a HuggingFace dataset
    """
    dataset = load_dataset("tokens", split)
    dataset = dataset.apply(lambda x: tokenize_prompt(tokenizer, x), axis=1)
    return Dataset.from_pandas(dataset)
