import torch 
from src.attention import MHA, spatial_attention, temporal_attention

torch.manual_seed(42)

x = torch.randn(2, 6, 32, dtype=torch.float32)

mha = MHA(C=32, H=4)

o, a = mha(x)

print("X:", x.shape)
print("O:", o.shape)
print("A:", a.shape)
print("attention row sums:")
print(a.sum(dim=-1))
print(a[0, 0])
print(a[0, 1])

mha1 = MHA(C=32, H=1)
mha4 = MHA(C=32, H=4)

o1, a1 = mha1(x)
o4, a4 = mha4(x)

print(a1.shape)
print(a4.shape)
print(o1.shape)
print(o4.shape)

B, T, J, C = 2, 9, 5, 32
H = 4

x = torch.randn(B, T, J, C)

spatial_mha = MHA(C, H)
temporal_mha = MHA(C, H)

Os, As = spatial_attention(x, spatial_mha)
Ot, At = temporal_attention(x, temporal_mha)

print("Spatial output:", Os.shape)
print("Spatial attention:", As.shape)

print("Temporal output:", Ot.shape)
print("Temporal attention:", At.shape)