import torch
from src.transformer import sinusoidal_pe, PreLNBlock, PostLNBlock, ZeroFFN, ZeroMHA
from src.attention import MHA

'''
'''
# pe = sinusoidal_pe(9, 32)

# print(pe.shape)        # [9, 32]
# print(pe[0])
# print(pe[1])

# B, L, C = 2, 9, 32

# x = torch.zeros(B, L, C)
# pe = sinusoidal_pe(L, C)

# y = x + pe

# print(y.shape)
# print(torch.allclose(y[0], pe))
# print(torch.allclose(y[1], pe))

# '''


# '''
# pre = PreLNBlock(32, 4)
# post = PostLNBlock(32, 4)

# pre.mha = ZeroMHA()
# pre.ffn = ZeroFFN()

# post.mha = ZeroMHA()
# post.ffn = ZeroFFN()

# torch.manual_seed(42)
# x = torch.randn(2, 6, 32)

# y_pre = pre(x)
# y_post = post(x)

# print("Pre-LN identity error:",
#       (y_pre - x).abs().max().item())

# print("Post-LN identity error:",
#       (y_post - x).abs().max().item())
# '''


# '''

# torch.manual_seed(42)

# x = torch.randn(1, 6, 32)

# mha = MHA(32, 4)
# mha.eval()

# perm = torch.tensor([2, 0, 5, 1, 4, 3])

# y = mha(x)

# x_perm = x[:, perm]
# y_prem = mha(x_perm)

# error = (
#     y_prem - y[:, perm]
# ).abs().max()

# print('without PE', error)

# pe = sinusoidal_pe(6, 32).unsqueeze(0)

# y_pe = mha(x + pe)

# x_perm = x[:,perm]
# y_prem_pe = mha(x_perm + pe)

# error_pe = (
#     y_prem_pe - y_pe[:, perm]
# ).abs().max()
# print('with PE', error_pe)

# '''


# '''

# torch.manual_seed(42)

# B, L, C, H = 2, 9, 32, 4

# x = torch.randn(
#     B, L, C,
#     requires_grad=True
# )

# pe = sinusoidal_pe(L, C)

# x_in = x + pe

# block = PreLNBlock(C, H)

# y = block(x_in)

# print("input:", x_in.shape)
# print("output:", y.shape)

# loss = y.square().mean()
# loss.backward()

# print("x grad finite:",
#       torch.isfinite(x.grad).all())

# print("output finite:",
#       torch.isfinite(y).all())
# '''

# '''

from src.transformer import SpatialBlock, TemporalBlock

x = torch.randn(4, 8, 16, 32)

B, T, J, C = x.shape

SB = SpatialBlock(C, 4)
TB = TemporalBlock(C, 4)

Xs, As = SB(x)
Xt, At = TB(x)

print(Xs.shape)
print(As.shape)
print(Xt.shape)
print(At.shape)