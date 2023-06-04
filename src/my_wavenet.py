import random
from typing import Final

from matplotlib import pyplot as plt
from lib.basic_tokenizer import make_tokenizers
import torch

from lib.list import split_list
from lib.nn.layers.BatchNorm1d import BatchNorm1d
from lib.nn.layers.Embedding import Embedding
from lib.nn.layers.FlattenConsecutive import FlattenConsecutive
from lib.nn.layers.Linear import Linear
from lib.nn.layers.Sequential import Sequential
from lib.nn.layers.Tanh import Tanh
from lib.nn.softmax import get_softmax


CTX_SIZE: Final[int] = 8  # must be 2**number of hidden layers
# EMBED_DIMS should be smaller than alphabet size to make sense
EMBED_DIMS: Final[int] = 12
HYPER_DIMS: Final[int] = 200
MINIBATCH_SIZE: Final[int] = 40
TRAINING_EPOCHS: Final[int] = 100000
LEARN_RATE_START: Final[float] = 0.2
LEARN_RATE_DECAY: Final[float] = 13


WORDS = open('sample_data/names.txt', 'r').read().splitlines()
random.seed(1234)
random.shuffle(WORDS)

ALPHABET = ['.'] + sorted(list(set(''.join(WORDS))))
ALPHABET_SIZE = len(ALPHABET)
(encode, decode) = make_tokenizers(ALPHABET)
ENCODED_WORDS = [encode(word + '.') for word in WORDS]


def make_samples(encoded_words: list[list[int]], context_size):
    xs, ys = [], []
    for encoded_word in encoded_words:
        context = [0] * context_size
        for encoded_char in encoded_word:
            xs.append(context)
            ys.append(encoded_char)
            context = context[1:] + [encoded_char]

    return xs, ys


def make_splits(list, split_sizes):
    splits = split_list(list, split_sizes)
    train, dev, test = [torch.tensor(split) for split in splits]
    return train, dev, test


# xs_list is a list of (lists with length CTX_SIZE)
xs_list, ys_list = make_samples(ENCODED_WORDS, CTX_SIZE)
xs_train, xs_dev, xs_test = make_splits(xs_list, [0.8, 0.1, 0.1])
ys_train, ys_dev, ys_test = make_splits(ys_list, [0.8, 0.1, 0.1])


model = Sequential([
    Embedding(ALPHABET_SIZE, EMBED_DIMS),
    FlattenConsecutive(2), Linear(2 * EMBED_DIMS, HYPER_DIMS,
                                  bias=False), BatchNorm1d(HYPER_DIMS), Tanh(),
    FlattenConsecutive(2), Linear(2 * HYPER_DIMS, HYPER_DIMS,
                                  bias=False), BatchNorm1d(HYPER_DIMS), Tanh(),
    FlattenConsecutive(2), Linear(2 * HYPER_DIMS, HYPER_DIMS,
                                  bias=False), BatchNorm1d(HYPER_DIMS), Tanh(),
    Linear(HYPER_DIMS, ALPHABET_SIZE),
])
parameters = model.parameters()

print("Parameter count", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True


def train(xs_train, ys_train):
    losses = []
    for i in range(TRAINING_EPOCHS):
        # minibatch
        idxs = torch.randint(0, xs_train.shape[0], (MINIBATCH_SIZE,))
        # forward pass
        logits = model(xs_train[idxs])
        loss = torch.nn.functional.cross_entropy(logits, ys_train[idxs])
        losses.append(loss.item())

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        LEARNING_RATE = (LEARN_RATE_START * LEARN_RATE_DECAY **
                         (-i/TRAINING_EPOCHS))
        for p in parameters:
            p.data -= LEARNING_RATE * p.grad  # type: ignore
        if i % (TRAINING_EPOCHS//10) == 0:
            print(f'{i}/{TRAINING_EPOCHS}', loss.item())

    print('Training loss', losses[-1])

    plt.plot(torch.tensor(losses).view(-1, 1000).mean(1))
    plt.show()


@torch.no_grad()
def evaluate_loss(xs, ys):
    logits = model(xs)
    loss = torch.nn.functional.cross_entropy(logits, ys)
    return loss


def generate_word(model, decode, context_size):
    encoded_chars = []
    context = [0] * context_size
    while True:
        logits = model(torch.tensor([context]))
        probabilities = get_softmax(logits, dim=1)
        current_char = int(torch.multinomial(
            probabilities, num_samples=1, replacement=True).item())
        context = context[1:] + [current_char]
        encoded_chars.append(current_char)
        if current_char == 0:
            break

    return decode(encoded_chars)


train(xs_train, ys_train)

# put layers into eval mode
for layer in model.layers:
    layer.training = False

# dev loss
dev_loss = evaluate_loss(xs_dev, ys_dev)
print('Dev loss', dev_loss.item())

print([generate_word(model, decode, CTX_SIZE) for _ in range(10)])
