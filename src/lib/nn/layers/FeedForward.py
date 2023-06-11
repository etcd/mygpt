import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, n_embed, n_dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            # projection for residual connection
            nn.Linear(4*n_embed, n_embed),
            # dropout
            nn.Dropout(n_dropout),
        )

    def forward(self, x):
        return self.layers(x)
