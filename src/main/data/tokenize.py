from typing import List

import torch
from transformers import CodeLlamaTokenizer

tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")


def tokenize(dataset: List[str]) -> List[torch.Tensor]:
    """
    Tokenizes a given set of inputs
    :param dataset:
    :return:
    """
    dataset_toks = []
    for query in dataset:
        dataset_toks.append(tokenizer.tokenize(query))
    return dataset_toks
