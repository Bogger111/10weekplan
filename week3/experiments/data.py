import torch

parents = torch.tensor([
    -1,  # root
    0,  # left_shoulder
    1,  # left_elbow
    2,  # left_wrist
    0,  # right_shoulder
    4,  # right_elbow
    5,  # right_wrist
])

offsets = torch.tensor([
    [ 0.0, 0.0, 0.0],
    [-0.5, 0.0, 0.0],
    [-0.7, 0.0, 0.0],
    [-0.6, 0.0, 0.0],
    [ 0.5, 0.0, 0.0],
    [ 0.7, 0.0, 0.0],
    [ 0.6, 0.0, 0.0],
], dtype=torch.float64)

R_local = torch.eye(
    3,
    dtype=torch.float64
).repeat(7, 1, 1)

root_trans = torch.zeros(
    3,
    dtype=torch.float64
)