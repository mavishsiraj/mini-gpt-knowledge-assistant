import torch

from feed_forward import FeedForward


X = torch.tensor([
    [0.4, -0.2, 0.7, 0.1],
    [0.6, -0.5, 0.3, 0.2],
    [0.2, -0.1, 0.8, 0.4]
])


ffn = FeedForward(
    embedding_dim=4,
    hidden_dim=16
)


output = ffn(X)


print("Input:")
print(X)

print("\nOutput:")
print(output)