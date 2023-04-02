
from typing import Callable
from torch import Tensor
import torch

from lib.bigrams.get_encoded_bigrams import get_encoded_bigrams


def calculate_likelihood(probabilities: Tensor, words: list[str], encode: Callable[[str], list[int]]):
    log_likelihood = 0.0
    n = 0
    for word in words:
        for c1, c2 in get_encoded_bigrams(word, encode):
            probability = probabilities[c1][c2]
            log_probability = torch.log(probability)
            log_likelihood += log_probability.item()
            n += 1
    return (log_likelihood, n)
