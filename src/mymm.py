import math
import torch
from lib.basic_tokenizer import make_tokenizers
from lib.bigrams.count import count_bigrams
from lib.bigrams.generate_word import generate_word
# from lib.bigrams.plot import plot_bigrams

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = sorted(list(set(''.join(words))))
(encode, decode) = make_tokenizers(alphabet, ['.'])

bigrams = count_bigrams(words, encode)
# plot_bigrams(bigrams, decode)

bigrams_plus_one = bigrams + 1  # model smoothing with +1
probabilities = bigrams_plus_one/bigrams_plus_one.sum(1, keepdim=True)

print([generate_word(probabilities, decode) for _ in range(5)])

log_likelihood = 0.0
n = 0
for word in words:
    chars = ['.'] + list(word) + ["."]
    encoded_chars = encode(chars)
    for c1, c2 in zip(encoded_chars, encoded_chars[1:]):
        probability = probabilities[c1][c2]
        log_probability = torch.log(probability)
        log_likelihood += log_probability.item()
        n += 1
print(f"Negative log likelihood: {-log_likelihood:.4f}")
print(f"Average negative log likelihood: {-log_likelihood/n:.4f}")
print(f"Perplexity: {math.exp(-log_likelihood/n):.4f}")
