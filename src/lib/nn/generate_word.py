from typing import Callable, Union
import torch

from lib.nn.softmax import get_softmax


def generate_word(
        neural_net: torch.Tensor,
        decode: Callable[[list[int]], str],
        generator: Union[torch.Generator, None] = None
):
    encoded_chars = []
    current_char = 0
    while True:
        encoded = torch.nn.functional.one_hot(
            torch.tensor([current_char]).to(torch.int64), num_classes=27).float()
        probabilities = get_softmax(encoded @ neural_net)
        current_char = int(torch.multinomial(
            probabilities, num_samples=1, replacement=True, generator=generator).item())
        encoded_chars.append(current_char)
        if current_char == 0:
            break
    return decode(encoded_chars)
