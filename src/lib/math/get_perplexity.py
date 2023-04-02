import math


def get_perplexity(log_likelihood: float, n: int):
    perplexity = math.exp(-log_likelihood/n)
    return perplexity
