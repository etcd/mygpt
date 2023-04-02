import torch

from lib.bigrams.get_bigrams import get_bigrams


def count_bigrams(encoded_words: list[list[int]], alphabet_size: int):
    bigram_counts = torch.zeros(
        (alphabet_size, alphabet_size), dtype=torch.int32)
    for encoded_word in encoded_words:
        for c1, c2 in get_bigrams(encoded_word):
            bigram_counts[c1][c2] += 1

    return bigram_counts
