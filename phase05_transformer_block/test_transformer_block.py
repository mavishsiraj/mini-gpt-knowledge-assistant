import torch

from phase05_transformer_block.transformer_block import TransformerBlock


X = torch.tensor([
    [0.2, 0.5, 0.1, 0.7],
    [0.4, 0.3, 0.8, 0.2],
    [0.9, 0.1, 0.6, 0.4]
])


block = TransformerBlock(
    embedding_dim=4,
    hidden_dim=16
)


output, attention_weights = block(X)


print("Input:")
print(X)

print("\nAttention Weights:")
print(attention_weights)

print("\nTransformer Block Output:")
print(output)