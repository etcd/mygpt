from typing import Callable


def get_encoded_bigrams(word: str, encode: Callable[[str], list[int]]):
    encoded = encode('.' + word + '.')
    return zip(encoded, encoded[1:])
