from typing import Callable
import torch


def count_bigrams(words: list[str], encode: Callable[[str], list[int]]):
    bigrams = torch.zeros((27, 27), dtype=torch.int32)
    for word in words:
        encoded = encode('.' + word + '.')
        for c1, c2 in zip(encoded, encoded[1:]):
            bigrams[c1][c2] += 1

    return bigrams
