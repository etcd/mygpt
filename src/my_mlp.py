from lib.basic_tokenizer import make_tokenizers
import torch


block_size = 3
embed_dims = 2
hyper_dims = 100

words = open('sample_data/names.txt', 'r').read().splitlines()
alphabet = ['.'] + sorted(list(set(''.join(words))))
alphabet_size = len(alphabet)
(encode, decode) = make_tokenizers(alphabet)
encoded_words = [encode(word + '.') for word in words]


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

embed_weights = torch.randn((alphabet_size, embed_dims))

hyper_weights = torch.randn((block_size*embed_dims, hyper_dims))
hyper_biases = torch.randn(hyper_dims)

out_weights = torch.randn((hyper_dims, alphabet_size))
out_biases = torch.randn(alphabet_size)

params = [embed_weights, hyper_weights, hyper_biases, out_weights, out_biases]

for p in params:
    p.requires_grad = True

print("Param count", sum(p.nelement() for p in params))

for _ in range(1000):
    # minibatch
    idxs = torch.randint(0, len(xs), (32,))

    # forward pass
    emb = embed_weights[xs[idxs]]  # (token count, block_size, embed_dims)
    h = torch.tanh(emb.view(-1, 6) @ hyper_weights + hyper_biases)
    logits = h @ out_weights + out_biases
    loss = torch.nn.functional.cross_entropy(logits, ys[idxs])
    # print(loss.item())

    # backward pass
    for p in params:
        p.grad = None
    loss.backward()

    # update
    for p in params:
        p.data -= 0.1 * p.grad  # type: ignore

# total loss
emb = embed_weights[xs]
h = torch.tanh(emb.view(-1, 6) @ hyper_weights + hyper_biases)
logits = h @ out_weights + out_biases
loss = torch.nn.functional.cross_entropy(logits, ys)
print(loss.item())
