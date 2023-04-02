import torch
import matplotlib.pyplot as plt
from lib.basic_tokenizer import make_decoder, make_encoder

words = open('sample_data/names.txt', 'r').read().splitlines()


alphabet = sorted(list(set(''.join(words))))
encode = make_encoder(alphabet, ['<S>', '<E>'])
decode = make_decoder(alphabet, ['<S>', '<E>'])

bigrams = torch.zeros((28, 28), dtype=torch.int32)
for word in words:
    chars = ['<S>'] + list(word) + ["<E>"]
    encoded_chars = encode(chars)
    for c1, c2 in zip(encoded_chars, encoded_chars[1:]):
        bigrams[c1][c2] += 1

plt.figure(figsize=(15, 15))
plt.imshow(bigrams, cmap="Blues")
for i in range(28):
    for j in range(28):
        plt.text(j, i, decode([i, j]),
                 ha='center', va='bottom', color="gray")
        plt.text(j, i, bigrams[i, j].item(),
                 ha='center', va='top', color="gray")
plt.axis('off')
plt.show()
