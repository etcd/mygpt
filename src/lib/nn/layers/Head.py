import torch
import torch.nn as nn


class Head(nn.Module):
    '''A self-attention head.'''

    def __init__(self, n_embed, block_size, head_size):
        super().__init__()
        # `Key` layer: what do I contain?
        # (B, T, Embeddings) -> (B, T, Head Size)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        # `Query` layer: what am I looking for?
        # (B, T, Embeddings) -> (B, T, Head Size)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        # `Value` layer: if affinities multiply by me, what do I output?
        # (B, T, Embeddings) -> (B, T, Head Size)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # constant, not a learnable parameter
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, E = x.shape
        keys = self.key(x)  # (B, T, Head Size)
        queries = self.query(x)  # (B, T, Head Size)

        # compute attention scores / affinities

        # (B, T, E) @ (B, E, T) -> (B, T, T)
        # for each sample in a batch, we compute the affinity matrix
        affinities = queries @ keys.transpose(-2, -1) * E**-0.5
        affinities = affinities.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        affinities = torch.nn.functional.softmax(
            affinities, dim=-1)  # (B, T, T)

        # perform weighted average of values
        values = self.value(x)  # (B, T, Head Size)
        # (B, T, T) @ (B, T, Head Size) -> (B, T, Head Size)
        out = affinities @ values
        return out


class MultiHeadAttention(nn.Module):
    '''Multiple heads of self-attention.'''

    def __init__(self, n_heads, n_embed, block_size):
        super().__init__()
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList([Head(n_embed, block_size, head_size)
                                    for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        # x is (B, T, E)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, E)
        out = self.proj(out)  # (B, T, E)
        return out
