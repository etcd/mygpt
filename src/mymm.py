import torch
from lib.basic_tokenizer import make_tokenizers
from lib.count_bigrams import count_bigrams
from lib.plot_bigrams import plot_bigrams

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = sorted(list(set(''.join(words))))
(encode, decode) = make_tokenizers(alphabet, ['.'])

bigrams = count_bigrams(words, encode)
plot_bigrams(bigrams, decode)

bigrams_plus_one = bigrams + 1  # model smoothing with +1
probabilities = bigrams_plus_one/bigrams_plus_one.sum(1, keepdim=True)

# generate a name


def generate_name(generator=None):
    encoded_out = []
    current_char = 0
    while True:
        current_char = torch.multinomial(
            probabilities[current_char], num_samples=1, replacement=True, generator=generator).item()
        encoded_out.append(current_char)
        if (current_char == 0):
            break

    return decode(encoded_out)


generator = torch.Generator().manual_seed(2147483647)
print([generate_name(generator) for _ in range(5)])

log_likelihood = 0
n = 0
for word in words:
    chars = ['.'] + list(word) + ["."]
    encoded_chars = encode(chars)
    for c1, c2 in zip(encoded_chars, encoded_chars[1:]):
        probability = probabilities[c1][c2]
        log_probability = torch.log(probability)
        log_likelihood += log_probability
        n += 1
print(f"Negative log likelihood: {-log_likelihood:.4f}")
print(f"Average negative log likelihood: {-log_likelihood/n:.4f}")
print(f"Perplexity: {torch.exp(-log_likelihood/n):.4f}")
