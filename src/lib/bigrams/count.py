from typing import Callable
import torch

from lib.bigrams.get_encoded_bigrams import get_encoded_bigrams


def count_bigrams(words: list[str], encode: Callable[[str], list[int]]):
    bigram_counts = torch.zeros((27, 27), dtype=torch.int32)
    for word in words:
        for c1, c2 in get_encoded_bigrams(word, encode):
            bigram_counts[c1][c2] += 1

    return bigram_counts
