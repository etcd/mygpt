import torch

from lib.basic_tokenizer import make_decoder, make_encoder

words = open('sample_data/names.txt', 'r').read().splitlines()

b = {}
for word in words:
    chars = ['<S>'] + list(word) + ["<E>"]
    for c1, c2 in zip(chars, chars[1:]):
        bigram = (c1, c2)
        b[bigram] = b.get(bigram, 0) + 1

print(sorted(b.items(), key=lambda kv: kv[1]))
N = torch.zeros((28, 28), dtype=torch.int32)

alphabet = sorted(list(set(''.join(words))))
encode = make_encoder(alphabet, ['<S>', '<E>'])
decode = make_decoder(alphabet, ['<S>', '<E>'])
