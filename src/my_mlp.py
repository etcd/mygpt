import random
from typing import Final

from matplotlib import pyplot as plt
from lib.basic_tokenizer import make_tokenizers
import torch

from lib.list import split_list
from lib.nn.softmax import get_softmax


CTX_SIZE: Final[int] = 3
EMBED_DIMS: Final[int] = 12
HYPER_DIMS: Final[int] = 200
MINIBATCH_SIZE: Final[int] = 38
TRAINING_EPOCHS: Final[int] = 250000
LEARN_RATE_START: Final[float] = 0.2
LEARN_RATE_DECAY: Final[float] = 10

words = open('sample_data/names.txt', 'r').read().splitlines()
random.seed(1234)
random.shuffle(words)

alphabet = ['.'] + sorted(list(set(''.join(words))))
alphabet_size = len(alphabet)
(encode, decode) = make_tokenizers(alphabet)
encoded_words = [encode(word + '.') for word in words]


def make_samples(encoded_words: list[list[int]]):
    xs, ys = [], []
    for encoded_word in encoded_words:
        context = [0] * CTX_SIZE
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
hyper_weights = torch.randn((CTX_SIZE*EMBED_DIMS, HYPER_DIMS))
hyper_biases = torch.randn(HYPER_DIMS)
out_weights = torch.randn((HYPER_DIMS, alphabet_size))
out_biases = torch.randn(alphabet_size)

params = [embed_weights, hyper_weights, hyper_biases, out_weights, out_biases]

for p in params:
    p.requires_grad = True

print("Param count", sum(p.nelement() for p in params))

losses = []
steps = range(TRAINING_EPOCHS)
for i in range(TRAINING_EPOCHS):
    # minibatch
    idxs = torch.randint(0, xs_train.shape[0], (MINIBATCH_SIZE,))
    batch = xs_train[idxs]
    labels = ys_train[idxs]

    # forward pass
    embedded_batch = embed_weights[batch]  # (batch size, ctx size, embed dims)
    hyper_pre_activate = embedded_batch.view(
        -1, CTX_SIZE * EMBED_DIMS) @ hyper_weights + hyper_biases
    hyper_activations = torch.tanh(hyper_pre_activate)
    logits = hyper_activations @ out_weights + out_biases
    loss = torch.nn.functional.cross_entropy(logits, labels)
    losses.append(loss.item())

    # backward pass
    for p in params:
        p.grad = None
    loss.backward()

    # update
    LEARNING_RATE = (LEARN_RATE_START * LEARN_RATE_DECAY **
                     (-i/TRAINING_EPOCHS))
    for p in params:
        p.data -= LEARNING_RATE * p.grad  # type: ignore

    if i % (TRAINING_EPOCHS//10) == 0:
        print(f'{i}/{TRAINING_EPOCHS}', losses[-1])

print('Training loss', losses[-1])

# plt.plot(steps, losses)
# plt.show()

# dev loss
embedded_batch = embed_weights[xs_dev]
hyper_pre_activate = embedded_batch.view(
    -1, CTX_SIZE*EMBED_DIMS) @ hyper_weights + hyper_biases
hyper_activations = torch.tanh(hyper_pre_activate)
logits = hyper_activations @ out_weights + out_biases
loss = torch.nn.functional.cross_entropy(logits, ys_dev)
print('Dev loss', loss.item())


def generate_word():
    encoded_chars = []
    context = [0] * CTX_SIZE
    while True:
        embedded = embed_weights[context]
        hyper_activations = torch.tanh(
            embedded.view(-1, CTX_SIZE*EMBED_DIMS)@hyper_weights + hyper_biases)
        logits = hyper_activations @ out_weights + out_biases
        probabilities = get_softmax(logits)
        current_char = int(torch.multinomial(
            probabilities, num_samples=1, replacement=True).item())
        context = context[1:] + [current_char]
        encoded_chars.append(current_char)
        if current_char == 0:
            break

    return decode(encoded_chars)


print([generate_word() for _ in range(10)])
