import torch
import torch.nn as nn

from phase03_attention.self_attention import SelfAttention
from phase04_feed_forward.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim
    ):
        super().__init__()

        # Self-Attention
        self.attention = SelfAttention(
            embedding_dim
        )

        # Feed Forward Network
        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, X):

        # -------------------------
        # 1. Normalize
        # -------------------------

        normalized_X = self.norm1(X)

        # -------------------------
        # 2. Self-Attention
        # -------------------------

        attention_output, attention_weights = \
            self.attention(normalized_X)

        # -------------------------
        # 3. Residual connection
        # -------------------------

        X = X + attention_output

        # -------------------------
        # 4. Normalize again
        # -------------------------

        normalized_X = self.norm2(X)

        # -------------------------
        # 5. Feed Forward
        # -------------------------

        ff_output = self.feed_forward(
            normalized_X
        )

        # -------------------------
        # 6. Second residual
        # -------------------------

        output = X + ff_output

        return output, attention_weights