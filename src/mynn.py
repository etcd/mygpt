from lib.basic_tokenizer import make_tokenizers
from lib.bigrams.get_bigrams import get_bigrams
import torch

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = sorted(list(set(''.join(words))))
(encode, decode) = make_tokenizers(alphabet, ['.'])

xs, ys = [], []
for word in words[:1]:
    encoded_word = encode('.' + word + '.')
    encoded_bigrams = get_bigrams(encoded_word)
    for c1, c2 in encoded_bigrams:
        xs.append(c1)
        ys.append(c2)

xs_tensor = torch.tensor(xs)
ys_tensor = torch.tensor(ys)

xenc = torch.nn.functional.one_hot(xs_tensor, num_classes=27).float()
print(xenc, xenc.shape)


W = torch.randn(27, 27)
logits = xenc @ W  # log counts
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
print(logits.exp())
