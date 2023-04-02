from typing import Callable
import torch


def generate_word(probabilities: torch.Tensor, decode: Callable[[list[int]], str], generator: torch.Generator = None):
    encoded_out: list[int] = []
    current_char = 0
    while True:
        current_char = torch.multinomial(
            probabilities[current_char], num_samples=1, replacement=True, generator=generator).item()
        encoded_out.append(current_char)
        if (current_char == 0):
            break

    return decode(encoded_out)
