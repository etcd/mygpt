import torch
import torch.nn as nn


class Head(nn.Module):
    '''A self-attention head.'''

    def __init__(self, n_embed, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)  # (B, T, C)
        query = self.query(x)  # (B, T, C)

        # compute attention scores / affinities

        # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = query @ key.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = torch.nn.functional.softmax(weights, dim=-1)  # (B, T, T)

        # perform weighted average of values
        value = self.value(x)  # (B, T, C)
        out = weights @ value  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    '''Multiple heads of self-attention.'''

    def __init__(self, num_heads, n_embed, block_size, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, block_size, head_size)
                                    for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
