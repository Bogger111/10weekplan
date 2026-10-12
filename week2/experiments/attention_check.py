import math
import torch
from src.attention import scaled_dot_product_attention

torch.manual_seed(42)

q = torch.randn(2, 4, 16, dtype=torch.float64)
k = torch.randn(2, 4, 16, dtype=torch.float64)
v = torch.randn(2, 4, 16, dtype=torch.float64)

raw_scores = q @ k.transpose(-2, -1)
scaled_scores = raw_scores / math.sqrt(q.shape[-1])

raw_attn = torch.softmax(raw_scores, dim=-1)
scaled_attn = torch.softmax(scaled_scores, dim=-1)

print(raw_attn.sum(-1))
print(scaled_attn.sum(-1))

# Q/K/V 已经投影并 split 成
# [N, H, L, dh]

heads = []

