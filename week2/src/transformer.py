import torch
import torch.nn as nn

from src.attention import MHA

import math

def sinusoidal_pe(L, C):
    pe = torch.zeros(L, C)

    for pos in range(L):
        for i in range(C // 2):
            pe[pos][2*i] = math.sin(float(pos / 10000 ** (2*i / C)))
            pe[pos][2*i+1] = math.cos(float(pos / 10000 ** (2*i / C)))

    return pe

class FFN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 2*C),
            nn.GELU(),
            nn.Linear(2*C, C)
        )
    def forward(self, x):
        return self.net(x)

class PreLNBlock(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

        self.mha = MHA(C, H)
        self.ffn = FFN(C)

    def forward(self, x):
        attn, A = self.mha(self.ln1(x))
        x = x + attn 
        x = x + self.ffn(self.ln2(x))

        return x, A

class PostLNBlock(nn.Module):
    def __init__(self, C, H):
        super().__init__()

        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

        self.mha = MHA(C, H)
        self.ffn = FFN(C)

    def forward(self, x):
        attn, A = self.mha(x)
        x = self.ln1(attn + x)
        x = self.ln2(self.ffn(x) + x)

        return x, A

class ZeroMHA(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x), None


class ZeroFFN(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class SpatialBlock(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.C = C
        self.H = H
        self.Block = PreLNBlock(C, H)

    def forward(self, x):
        B, T, J, C = x.shape
        x = x.reshape(B*T, J, C)
        x, attn = self.Block(x)

        x = x.reshape(B, T, J, C)
        attn = attn.reshape(B, T, self.H, J, J)

        return x, attn

class TemporalBlock(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.C = C
        self.H = H
        self.Block = PreLNBlock(C, H)

    def forward(self, x):
        B, T, J, C = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(B*J, T, C)
        x, attn = self.Block(x)

        x = x.reshape(B, J, T, C).transpose(1, 2)
        attn = attn.reshape(B, J, self.H, T, T)

        return x, attn
        