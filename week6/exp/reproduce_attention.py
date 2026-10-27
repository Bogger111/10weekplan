import torch
import math

qkv = torch.randn(1, 100, 1536)

def split_heads(x: torch.Tensor, num_heads=16):
    B, N, D = x.shape
    assert D % num_heads == 0

    return x.reshape(
        B, N, num_heads, D // num_heads
    ).permute(0, 2, 1, 3)


q, k, v = qkv.chunk(3, dim=-1)

q = split_heads(q)
k = split_heads(k)
v = split_heads(v)

scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])

attn = torch.softmax(scores, dim=-1)

o = attn @ v

print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)
print("Scores:", scores.shape)
print("Attention:", attn.shape)
print("Output:", o.shape)