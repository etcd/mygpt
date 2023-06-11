
import random
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
BATCH_SIZE = 4  # number of sequences to process in parallel
EMBED_DIMS = 32  # embedding dimensions
TRAINING_STEPS = 20000
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dims):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, embed_dims)  # (V, E)
        self.position_embedding_table = nn.Embedding(
            BLOCK_SIZE, embed_dims)  # (T, E)
        self.sa_head = Head(embed_dims, BLOCK_SIZE, embed_dims)
        self.lm_head = nn.Linear(embed_dims, vocab_size)

    def forward(self, context, targets=None):
        # context is (B, T)
        token_embeddings = self.token_embedding_table(context)  # (B, T, E)
        position_embeddings = self.position_embedding_table(
            torch.arange(context.shape[1], device=DEVICE))  # (T, E)
        x = token_embeddings + position_embeddings  # (B, T, E)
        x = self.sa_head(x)  # apply one head of self attention (B, T, E)
        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, E = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B*T, E), targets.reshape(B*T))
        return logits, loss

    def generate(self, context, max_new_tokens):
        # context is (B, T)
        for _ in range(max_new_tokens):
            context_crop = context[:, -BLOCK_SIZE:]  # (B, T)
            token_embeddings = self.token_embedding_table(
                context_crop)  # (B, T, E)
            position_embeddings = self.position_embedding_table(
                torch.arange(context_crop.shape[1], device=DEVICE))  # (T, E)
            x = token_embeddings + position_embeddings  # (B, T, E)
            x = self.sa_head(x)
            logits = self.lm_head(x)  # (B, T, V)
            logits = logits[:, -1, :]  # get last time step; (B, V)
            probabilities = torch.nn.functional.softmax(
                logits, dim=-1)  # (B, V)
            context_next = torch.multinomial(
                probabilities, num_samples=1)  # (B, 1)
            context = torch.cat([context, context_next], dim=1)  # (B, T+1)
        return context


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


xs, ys = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)

model = LanguageModel(len(ALPHABET), EMBED_DIMS)
model = model.to(DEVICE)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for steps in range(TRAINING_STEPS):
    xs, ys = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
    logits, loss = model(xs, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
