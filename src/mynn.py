from lib.basic_tokenizer import make_tokenizers
from lib.bigrams.get_bigrams import get_bigrams
import torch
from lib.nn.generate_word import generate_word

from lib.nn.softmax import get_softmax

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = ['.'] + sorted(list(set(''.join(words))))
alphabet_size = len(alphabet)
(encode, decode) = make_tokenizers(alphabet)
encoded_words = [encode('.' + word + '.') for word in words]


def make_samples(encoded_words: list[list[int]]):
    xs, ys = [], []
    for encoded_word in encoded_words:
        encoded_bigrams = get_bigrams(encoded_word)
        for c1, c2 in encoded_bigrams:
            xs.append(c1)
            ys.append(c2)
    return xs, ys


xs, ys = make_samples(encoded_words)

xs_tensor = torch.tensor(xs)
ys_tensor = torch.tensor(ys)

xenc = torch.nn.functional.one_hot(
    xs_tensor, num_classes=alphabet_size).float()

neural_net = torch.randn((alphabet_size, alphabet_size), requires_grad=True)

# gradient descent
for i in range(30):
    # forward pass
    next_letter_probabilities = get_softmax(xenc @ neural_net)
    loss = -next_letter_probabilities[range(len(ys)), ys].log().mean()
    # print(loss.item())

    # backward pass
    neural_net.grad = None
    loss.backward()

    # update weights
    neural_net.data += -50 * neural_net.grad  # type: ignore

for _ in range(5):
    print(generate_word(neural_net, decode, alphabet_size))
