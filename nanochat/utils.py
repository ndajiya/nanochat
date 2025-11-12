import torch

def apply_rotary_pos_emb(x, cos, sin):
    # x: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    # We want to apply the rotary embeddings to the last dimension (head_dim)
    # and broadcast across batch and num_heads
    return (x * cos) + (rotate_half(x) * sin)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)