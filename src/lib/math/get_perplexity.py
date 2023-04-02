import math


def get_perplexity(log_likelihood: float, num_bigrams: int):
    perplexity = math.exp(-log_likelihood/num_bigrams)
    return perplexity
