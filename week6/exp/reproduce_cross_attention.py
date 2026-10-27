import torch
import math
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)

queries = torch.randn(1, 100, 1024)
context = torch.randn(1, 1536, 1024)

to_q = nn.Linear(1024, 512)
to_kv = nn.Linear(1024, 1024)
to_out = nn.Linear(512, 1024)


def split_heads(x, num_heads=16):
    B, N, D = x.shape
    assert D % num_heads == 0

    return x.reshape(
        B, N, num_heads, D // num_heads
    ).transpose(1, 2)


q = to_q(queries)
kv = to_kv(context)

k, v = kv.chunk(2, dim=-1)

q = split_heads(q)
k = split_heads(k)
v = split_heads(v)

print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)

scores = q @ k.transpose(-1, -2)
scores = scores / math.sqrt(q.shape[-1])

print("Scores:", scores.shape)

attn = torch.softmax(scores, dim=-1)

print("Attention:", attn.shape)

o = attn @ v

print("Weighted output:", o.shape)

B, H, N, D = o.shape

o = o.transpose(1, 2).contiguous()
o = o.reshape(B, N, H * D)

print("Merged output:", o.shape)

output = to_out(o)

print("Final output:", output.shape)


# ===== Numerical Validation =====

# Test 1: Attention 权重归一化
attn_sum = attn.sum(dim=-1)

assert torch.allclose(
    attn_sum,
    torch.ones_like(attn_sum),
    atol=1e-5
)

print("[PASS] Attention normalization")


# Test 2: 与 PyTorch 官方 SDPA 比较
reference = F.scaled_dot_product_attention(q, k, v)

max_error = (reference - (attn @ v)).abs().max().item()

print(f"Max numerical difference: {max_error:.8f}")

assert torch.allclose(
    reference,
    attn @ v,
    atol=1e-5,
    rtol=1e-5
)

print("[PASS] PyTorch SDPA comparison")


# Test 3: 最终输出 Shape
assert output.shape == (1, 100, 1024)

print("[PASS] Final output shape")