import math
from typing import TypeVar

X = TypeVar("X")


def split_list(list: list[X], fractions: list[float]) -> list[list[X]]:
    """Split a list into multiple lists based on fractions.
    """
    if sum(fractions) != 1:
        raise ValueError("Fractions must sum to 1.")

    start = 0
    result = []
    for fraction in fractions:
        end = start + math.floor(len(list)*fraction)
        result.append(list[start:end])
        start = end
    return result
