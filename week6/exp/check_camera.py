import torch

def perspective_projection(points, K):
    uvw = points @ K.T

    uv = uvw[..., :2] / uvw[..., 2:3]

    return uv


K = torch.tensor([
    [500., 0., 320.],
    [0., 500., 240.],
    [0., 0., 1.]
])

points = torch.tensor([[
    [1., 0., 5.],
    [1., 0., 10.]
]])

print(perspective_projection(points, K))

