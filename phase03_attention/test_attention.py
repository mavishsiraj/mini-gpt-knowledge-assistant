import torch

from self_attention import SelfAttention


# 3 tokens
# embedding dimension = 4

X = torch.tensor([
    [0.2, 0.5, 0.1, 0.7],
    [0.4, 0.3, 0.8, 0.2],
    [0.9, 0.1, 0.6, 0.4]
])


attention = SelfAttention(
    embedding_dim=4
)


output, attention_weights = attention(X)


print("Input X:")
print(X)

print("\nAttention Weights:")
print(attention_weights)

print("\nAttention Output:")
print(output)