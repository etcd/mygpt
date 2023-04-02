import torch
import matplotlib.pyplot as plt
from lib.basic_tokenizer import make_tokenizers

words = open('sample_data/names.txt', 'r').read().splitlines()


alphabet = sorted(list(set(''.join(words))))
[encode, decode] = make_tokenizers(alphabet, ['.'])


bigrams = torch.zeros((27, 27), dtype=torch.int32)
for word in words:
    chars = ['.'] + list(word) + ["."]
    encoded_chars = encode(chars)
    for c1, c2 in zip(encoded_chars, encoded_chars[1:]):
        bigrams[c1][c2] += 1

# plot the bigrams

# plt.figure(figsize=(15, 15))
# plt.imshow(bigrams, cmap="Blues")
# for i in range(27):
#     for j in range(27):
#         plt.text(j, i, decode([i, j]),
#                  ha='center', va='bottom', color="gray")
#         plt.text(j, i, bigrams[i, j].item(),
#                  ha='center', va='top', color="gray")
# plt.axis('off')
# plt.show()

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
