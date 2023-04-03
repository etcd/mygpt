from lib.basic_tokenizer import make_tokenizers
from lib.bigrams.get_bigrams import get_bigrams
import torch
from lib.nn.generate_word import generate_word

from lib.nn.softmax import get_softmax

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = ['.'] + sorted(list(set(''.join(words))))
alphabet_size = len(alphabet)
(encode, decode) = make_tokenizers(alphabet)
encoded_words = [encode(word + '.') for word in words]

block_size = 3
embedding_dims = 2
hyper_dims = 200


def make_samples(encoded_words: list[list[int]]):
    xs, ys = [], []
    for encoded_word in encoded_words:
        context = [0] * block_size
        for c_enc in encoded_word:
            xs.append(context)
            ys.append(c_enc)
            context = context[1:] + [c_enc]

    return xs, ys


xs_list, ys_list = make_samples(encoded_words)

xs = torch.tensor(xs_list)
ys = torch.tensor(ys_list)

embedding_map = torch.randn(
    (alphabet_size, embedding_dims), requires_grad=True)

hyper_weights = torch.randn(
    (block_size*embedding_dims, hyper_dims), requires_grad=True)
hyper_biases = torch.randn(hyper_dims, requires_grad=True)

hyper_weights2 = torch.randn((hyper_dims, alphabet_size), requires_grad=True)
hyper_biases2 = torch.randn(alphabet_size, requires_grad=True)

parameters = [embedding_map, hyper_weights,
              hyper_biases, hyper_weights2, hyper_biases2]

parameter_count = sum(p.nelement() for p in parameters)
print("Param count", parameter_count)

for _ in range(1000):
    # forward pass
    emb = embedding_map[xs]  # (tokens in words, block_size, embedding_dims)
    h = torch.tanh(emb.view(-1, 6) @ hyper_weights + hyper_biases)
    logits = h @ hyper_weights2 + hyper_biases2
    loss = torch.nn.functional.cross_entropy(logits, ys)
    print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    for p in parameters:
        p.data -= 0.1 * p.grad  # type: ignore
