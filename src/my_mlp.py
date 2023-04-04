from typing import Final
from lib.basic_tokenizer import make_tokenizers
import torch

from lib.list import split_list


BLOCK_SIZE: Final[int] = 3
EMBED_DIMS: Final[int] = 10
HYPER_DIMS: Final[int] = 100
MINIBATCH_SIZE: Final[int] = 32
TRAINING_EPOCHS: Final[int] = 50000
LEARNING_RATE: Final[float] = 0.1

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = ['.'] + sorted(list(set(''.join(words))))
alphabet_size = len(alphabet)
(encode, decode) = make_tokenizers(alphabet)
encoded_words = [encode(word + '.') for word in words]


def make_samples(encoded_words: list[list[int]]):
    xs, ys = [], []
    for encoded_word in encoded_words:
        context = [0] * BLOCK_SIZE
        for c_enc in encoded_word:
            xs.append(context)
            ys.append(c_enc)
            context = context[1:] + [c_enc]

    return xs, ys


xs_list, ys_list = make_samples(encoded_words)
xs_split = split_list(xs_list, [0.8, 0.1, 0.1])  # train, dev, test
ys_split = split_list(ys_list, [0.8, 0.1, 0.1])  # train, dev, test
xs_train, xs_dev, xs_test = [torch.tensor(l) for l in xs_split]
ys_train, ys_dev, ys_test = [torch.tensor(l) for l in ys_split]

embed_weights = torch.randn((alphabet_size, EMBED_DIMS))

hyper_weights = torch.randn((BLOCK_SIZE*EMBED_DIMS, HYPER_DIMS))
hyper_biases = torch.randn(HYPER_DIMS)

out_weights = torch.randn((HYPER_DIMS, alphabet_size))
out_biases = torch.randn(alphabet_size)

params = [embed_weights, hyper_weights, hyper_biases, out_weights, out_biases]

for p in params:
    p.requires_grad = True

print("Param count", sum(p.nelement() for p in params))

for _ in range(TRAINING_EPOCHS):
    # minibatch
    idxs = torch.randint(0, len(xs_train), (MINIBATCH_SIZE,))

    # forward pass
    # (token count, block_size, embed_dims)
    emb = embed_weights[xs_train[idxs]]
    h = torch.tanh(emb.view(-1, BLOCK_SIZE*EMBED_DIMS)
                   @ hyper_weights + hyper_biases)
    logits = h @ out_weights + out_biases
    loss = torch.nn.functional.cross_entropy(logits, ys_train[idxs])
    # print(loss.item())

    # backward pass
    for p in params:
        p.grad = None
    loss.backward()

    # update
    for p in params:
        p.data -= LEARNING_RATE * p.grad  # type: ignore

# dev loss
emb = embed_weights[xs_dev]
h = torch.tanh(emb.view(-1, BLOCK_SIZE*EMBED_DIMS)
               @ hyper_weights + hyper_biases)
logits = h @ out_weights + out_biases
loss = torch.nn.functional.cross_entropy(logits, ys_dev)
print(loss.item())
