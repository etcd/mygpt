def get_bigrams(encoded_word: list[int]):
    return zip(encoded_word, encoded_word[1:])
