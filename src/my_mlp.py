import random
from typing import Final

from matplotlib import pyplot as plt
from lib.basic_tokenizer import make_tokenizers
import torch

from lib.list import split_list
from lib.nn.softmax import get_softmax


CTX_SIZE: Final[int] = 4
EMBED_DIMS: Final[int] = 12
HYPER_DIMS: Final[int] = 200
MINIBATCH_SIZE: Final[int] = 40
TRAINING_EPOCHS: Final[int] = 100000
LEARN_RATE_START: Final[float] = 0.2
LEARN_RATE_DECAY: Final[float] = 13


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
        for encoded_char in encoded_word:
            xs.append(context)
            ys.append(encoded_char)
            context = context[1:] + [encoded_char]

    return xs, ys


xs_list, ys_list = make_samples(encoded_words)
xs_split = split_list(xs_list, [0.8, 0.1, 0.1])  # train, dev, test
ys_split = split_list(ys_list, [0.8, 0.1, 0.1])  # train, dev, test
xs_train, xs_dev, xs_test = [torch.tensor(l) for l in xs_split]
ys_train, ys_dev, ys_test = [torch.tensor(l) for l in ys_split]

embed_weights = torch.randn((alphabet_size, EMBED_DIMS))
hyper_weights = torch.randn(
    (CTX_SIZE*EMBED_DIMS, HYPER_DIMS)) * (5/3)/(CTX_SIZE*EMBED_DIMS)**.5
out_weights = torch.randn((HYPER_DIMS, alphabet_size)) * (5/3)/(HYPER_DIMS)**.5
out_biases = torch.randn(alphabet_size) * 0.01

batchnorm_gains = torch.ones((1, HYPER_DIMS))
batchnorm_biases = torch.zeros((1, HYPER_DIMS))

params = [embed_weights, hyper_weights, out_weights,
          out_biases, batchnorm_gains, batchnorm_biases]

for p in params:
    p.requires_grad = True

print("Param count", sum(p.nelement() for p in params))


def evaluate_loss(ins: torch.Tensor, outs: torch.Tensor):
    embedded = embed_weights[ins]  # (ins size, ctx size, embed dims)
    hyper_pre_activate = embedded.view(
        -1, CTX_SIZE * EMBED_DIMS) @ hyper_weights

    # batch norm
    batchnorm_mean = hyper_pre_activate.mean(0,  keepdim=True)
    batchnorm_std = hyper_pre_activate.std(0, keepdim=True)
    hyper_pre_activate = batchnorm_gains * \
        (hyper_pre_activate - batchnorm_mean) / \
        (batchnorm_std + 1e-100) + batchnorm_biases

    hyper_activations = torch.tanh(hyper_pre_activate)
    logits = hyper_activations @ out_weights + out_biases  # log counts

    # plt.figure(figsize=(20, 10))
    # plt.imshow(hyper_activations.abs() > 0.99,  # type: ignore
    #            cmap='gray', interpolation='nearest')
    # plt.show()

    return torch.nn.functional.cross_entropy(logits, outs), batchnorm_mean, batchnorm_std


losses = []
steps = range(TRAINING_EPOCHS)
batchnorm_mean_running = torch.zeros((1, HYPER_DIMS))
batchnorm_std_running = torch.ones((1, HYPER_DIMS))
for i in range(TRAINING_EPOCHS):
    # minibatch
    idxs = torch.randint(0, xs_train.shape[0], (MINIBATCH_SIZE,))

    # forward pass
    loss, batchnorm_mean_i, batchnorm_std_i = evaluate_loss(
        xs_train[idxs], ys_train[idxs])
    losses.append(loss.item())

    with torch.no_grad():
        batchnorm_mean_running = .999*batchnorm_mean_running+0.001*batchnorm_mean_i
        batchnorm_std_running = .999*batchnorm_std_running+0.001*batchnorm_std_i

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
        print(f'{i}/{TRAINING_EPOCHS}', loss.item())

print('Training loss', losses[-1])

plt.plot(steps, losses)
plt.show()

# dev loss
loss, _, _ = evaluate_loss(xs_dev, ys_dev)
print('Dev loss', loss.item())


def generate_word():
    encoded_chars = []
    context = [0] * CTX_SIZE
    while True:
        embedded = embed_weights[context]
        hyper_pre_activate = embedded.view(-1,
                                           CTX_SIZE * EMBED_DIMS) @ hyper_weights

        # batch norm
        hyper_pre_activate = batchnorm_gains * \
            (hyper_pre_activate - batchnorm_mean_running) / \
            (batchnorm_std_running + 1e-100) + batchnorm_biases

        hyper_activations = torch.tanh(hyper_pre_activate)
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
