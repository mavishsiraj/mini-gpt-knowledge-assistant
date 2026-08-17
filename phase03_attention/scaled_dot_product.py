import torch


def scaled_dot_product_attention(Q, K, V):

    # Step 1:
    # Query aur Key ka dot product
    scores = Q @ K.T

    # Step 2:
    # Scale by sqrt(d_k)
    d_k = K.shape[-1]

    scaled_scores = scores / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )

    # Step 3:
    # Convert scores into probabilities
    attention_weights = torch.softmax(
        scaled_scores,
        dim=-1
    )

    # Step 4:
    # Weighted combination of Values
    output = attention_weights @ V

    return output, attention_weights