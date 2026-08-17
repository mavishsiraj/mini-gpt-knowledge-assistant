import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        # Learnable matrices
        self.Wq = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.Wk = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.Wv = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, X):

        # -------------------------
        # 1. Create Q, K, V
        # -------------------------

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        # -------------------------
        # 2. Calculate scores
        # -------------------------

        scores = Q @ K.T

        # -------------------------
        # 3. Scale
        # -------------------------

        d_k = K.shape[-1]

        scores = scores / torch.sqrt(
            torch.tensor(
                d_k,
                dtype=torch.float32
            )
        )

        # -------------------------
        # 4. Causal mask
        # -------------------------

        seq_len = X.shape[0]

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # -------------------------
        # 5. Softmax
        # -------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -------------------------
        # 6. Weighted values
        # -------------------------

        output = attention_weights @ V

        return output, attention_weights