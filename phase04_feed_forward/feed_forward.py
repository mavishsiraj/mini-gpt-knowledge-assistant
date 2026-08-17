import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(

            # Expand
            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            # Non-linearity
            nn.GELU(),

            # Compress
            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )

    def forward(self, X):

        return self.network(X)