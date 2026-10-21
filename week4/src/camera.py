import torch
def to_camera(points, R, t):
    points_cam = (R @ points.transpose(-1, -2)).transpose(-1, -2) + t
    return points_cam

def project_perspective(points_cam, K, z_min=1e-4):
    points_t = points_cam @ K.T
    uv = points_t[:, :2] / points_t[:, 2:3]

    finite = torch.isfinite(points_cam).all(dim=-1)
    positive_depth = points_cam[:, 2] > z_min

    valid = finite & positive_depth

    return uv, valid