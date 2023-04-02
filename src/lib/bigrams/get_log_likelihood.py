from torch import Tensor
import torch

from lib.bigrams.get_bigrams import get_bigrams


def get_log_likelihood(probabilities: Tensor, encoded_word: list[int]):
    encoded_bigrams = get_bigrams(encoded_word)
    encoded_probabilities = [probabilities[c1][c2]
                             for c1, c2 in encoded_bigrams]
    log_likelihoods = [torch.log(p).item() for p in encoded_probabilities]

    return sum(log_likelihoods)
