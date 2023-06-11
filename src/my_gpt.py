
import random
import time
import torch
import torch.nn as nn

from lib.basic_tokenizer import make_tokenizers
from lib.list import split_list
from lib.nn.layers.FeedForward import FeedForward
from lib.nn.layers.Head import Head, MultiHeadAttention

'''
B = Batch
T = Time
V = Vocabulary size (alphabet size)
E = Embedding dimensions
'''


torch.manual_seed(1234)

BLOCK_SIZE = 32  # max context length for predictions
BATCH_SIZE = 32  # number of sequences to process in parallel
N_EMBED = 64  # embedding dimensions
NUM_HEADS = 4  # number of heads in multi-head attention
N_LAYERS = 4  # number of transformer blocks
TRAINING_STEPS = 10000
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Block(nn.Module):
    '''Transformer block: communication followed by computation.'''

    def __init__(self, n_heads, n_embed, block_size):
        super().__init__()
        # communication via multi-headed self-attention
        self.self_attention = MultiHeadAttention(
            n_heads, n_embed, block_size)
        # computation via feed-forward network
        self.ffwd = FeedForward(n_embed)
        # layer normalization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residual connections achieved with `x +`
        # communication
        x = x + self.self_attention(self.ln1(x))  # (B, T, E)
        # computation
        x = x + self.ffwd(self.ln2(x))  # (B, T, E)
        return x


class DecoderTransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embed)  # (V, E)
        self.position_embedding_table = nn.Embedding(
            BLOCK_SIZE, n_embed)  # (T, E)
        self.blocks = nn.Sequential(
            *[Block(NUM_HEADS, n_embed, BLOCK_SIZE) for _ in range(N_LAYERS)]
        )
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, xs, targets=None):
        # xs is (B, T)
        token_embeddings = self.token_embedding_table(xs)  # (B, T, E)
        position_embeddings = self.position_embedding_table(
            torch.arange(xs.shape[1], device=DEVICE))  # (T, E)
        x = token_embeddings + position_embeddings  # (B, T, E)
        x = self.blocks(x)  # (B, T, E)
        x = self.ln_final(x)  # (B, T, E)
        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, E = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B*T, E), targets.reshape(B*T))
        return logits, loss

    def generate(self, xs, max_new_tokens):
        # xs is (B, T)
        for _ in range(max_new_tokens):
            xs_crop = xs[:, -BLOCK_SIZE:]  # (B, T)
            logits, _ = self(xs_crop)  # (B, T, V)

            logits = logits[:, -1, :]  # get last time step; (B, V)
            probabilities = torch.nn.functional.softmax(
                logits, dim=-1)  # (B, V)
            xs_next = torch.multinomial(probabilities, num_samples=1)  # (B, 1)
            xs = torch.cat([xs, xs_next], dim=1)  # (B, T+1)
        return xs


def make_sample(data, index, block_size):
    '''Makes a single training _sample_, which has a time dimension.

       For some time `t`, we have a sub-sample, where:
            training input: xs[0, t+1]
            training target: ys[t]
    '''
    xs = data[index:index+block_size]
    ys = data[index+1:block_size+index+1]
    return xs, ys


def get_batch(data, block_size, batch_size):
    '''Gets a random _batch_ of _samples_ from data.
    '''

    # get `batch_size` number of random indices (from 0 to `len(data)-block_size`)
    indices = [random.randint(0, len(data)-block_size)
               for _ in range(batch_size)]

    # make samples from data at those indices, turn into tensors, split into xs and ys
    xs, ys = torch.split(torch.stack(
        [torch.stack(make_sample(data, i, block_size)) for i in indices]), 1, dim=1)

    xs = xs.squeeze(1).to(DEVICE)
    ys = ys.squeeze(1).to(DEVICE)

    return xs, ys


TEXT = open('sample_data/tinyshakespeare.txt', 'r').read()
ALPHABET = sorted(list(set(TEXT)))
(encode, decode) = make_tokenizers(ALPHABET)

DATA = torch.tensor(encode(TEXT), dtype=torch.long)
train_data, validate_data = split_list(DATA, [0.9, 0.1])

model = DecoderTransformerModel(len(ALPHABET), N_EMBED)
model = model.to(DEVICE)

print("Parameter count", sum(p.nelement() for p in model.parameters()))

start_time = time.time()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
for steps in range(TRAINING_STEPS):
    xs, ys = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
    logits, loss = model(xs, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training time", time.time() - start_time)
print("Loss", loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
