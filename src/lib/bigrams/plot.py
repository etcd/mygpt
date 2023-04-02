from typing import Callable
from matplotlib import pyplot as plt
from torch import Tensor


def plot_bigrams(encoded_bigrams: Tensor, decode: Callable[[list[int]], str]):
    plt.figure(figsize=(15, 15))
    plt.imshow(encoded_bigrams, cmap="Blues")  # type: ignore
    for x in range(27):
        for y in range(27):
            plt.text(x, y, decode([y, x]),  # type: ignore
                     ha='center', va='bottom', color="gray")
            plt.text(x, y, encoded_bigrams[y, x].item(),  # type: ignore
                     ha='center', va='top', color="gray")
    plt.axis('off')  # type: ignore
    plt.show()
