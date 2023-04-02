from lib.basic_tokenizer import make_tokenizers
from lib.bigrams.get_bigrams import get_bigrams
import torch

from lib.nn.softmax import get_softmax

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = sorted(list(set(''.join(words))))
(encode, decode) = make_tokenizers(alphabet, ['.'])

xs, ys = [], []
for word in words:
    encoded_word = encode('.' + word + '.')
    encoded_bigrams = get_bigrams(encoded_word)
    for c1, c2 in encoded_bigrams:
        xs.append(c1)
        ys.append(c2)

xs_tensor = torch.tensor(xs)
ys_tensor = torch.tensor(ys)

xenc = torch.nn.functional.one_hot(xs_tensor, num_classes=27).float()

neural_net = torch.randn((27, 27), requires_grad=True)

# gradient descent
for i in range(100):
    # forward pass
    next_letter_probabilities = get_softmax(xenc @ neural_net)
    loss = -next_letter_probabilities[range(len(ys)), ys].log().mean()
    print(loss.item())

    # backward pass
    neural_net.grad = None
    loss.backward()

    # update weights
    neural_net.data += -50 * neural_net.grad  # type: ignore
