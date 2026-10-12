import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(
    q,
    k,
    v,
    blocked_mask=None
):
    assert q.shape[-1] == k.shape[-1], "Mismatch q, k"
    assert k.shape[-2] == v.shape[-2], "Mismatch k, v"

    dk = q.shape[-1]

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)

    if blocked_mask is not None:
        if blocked_mask.dtype != torch.bool:
            raise TypeError("blocked_mask must be bool")

        if blocked_mask.all(dim=-1).any():
            raise ValueError("A query cannot have all keys blocked")

        scores = scores.masked_fill(blocked_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)

    output = attn @ v

    if not torch.isfinite(attn).all():
        raise ValueError("Attention contains NaN or Inf")

    if not torch.isfinite(output).all():
        raise ValueError("Output contains NaN or Inf")

    return output,attn


class MHA(nn.Module):
    def __init__(self, C, H):
        super().__init__()

        assert C % H == 0, "C must be divisible by H"

        self.C = C
        self.H = H
        self.dh = C // H

        self.wq = nn.Linear(C, C)
        self.wk = nn.Linear(C, C)
        self.wv = nn.Linear(C, C)
        self.wo = nn.Linear(C, C)

    def forward(self, X, blocked_mask=None):
        # X: [N, L, C]
        N, L, C = X.shape

        assert C == self.C

        Q = self.wq(X)  # [N, L, C]
        K = self.wk(X)  # [N, L, C]
        V = self.wv(X)  # [N, L, C]

        Q = Q.reshape(N, L, self.H, self.dh).transpose(1, 2)
        K = K.reshape(N, L, self.H, self.dh).transpose(1, 2)
        V = V.reshape(N, L, self.H, self.dh).transpose(1, 2)

        O, A = scaled_dot_product_attention(
            Q, K, V,
            blocked_mask=blocked_mask
        )

        O = O.transpose(1, 2).contiguous()
        O = O.reshape(N, L, C)

        O = self.wo(O)

        return O, A


def spatial_attention(x, mha):
        # x;[B, T, J, C]
        B, T, J, C = x.shape
        x = x.reshape(B*T, J, C)
        O , A = mha(x)
        return O.reshape(B, T, J, C), A

def temporal_attention(x, mha):
        B, T, J, C = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(B*J, T, C)
        O , A = mha(x)
        O = O.reshape(B, J, T, C)
        return O.transpose(1, 2), A

