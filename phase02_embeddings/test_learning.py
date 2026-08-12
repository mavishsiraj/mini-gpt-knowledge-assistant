import torch
import torch.nn as nn
import torch.optim as optim


embedding = nn.Embedding(3, 4)

optimizer = optim.SGD(
    embedding.parameters(),
    lr=0.1
)


print("BEFORE:")
print(embedding.weight)


# We want token 0's embedding
# to become close to this target vector.

target = torch.tensor([
    1.0,
    1.0,
    1.0,
    1.0
])


for step in range(100):

    optimizer.zero_grad()

    vector = embedding(
        torch.tensor([0])
    )[0]

    loss = ((vector - target) ** 2).mean()

    loss.backward()

    optimizer.step()


print("\nAFTER:")
print(embedding.weight)
print("\nRequires gradient:")
print(embedding.weight.requires_grad)