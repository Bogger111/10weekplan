import torch
def to_camera(points, R, t):
    points_cam = torch.matmul(R, points.transpose(-1, -2)).transpose(-1, -2) + t.reshape(*points.shape[:-2], 1, 3)
    return points_cam

def project_perspective(points_cam, K, z_min=1e-4):
    points_t = torch.matmul(points_cam, K.transpose(-1, -2))
    finite = torch.isfinite(points_cam).all(dim=-1)
    positive_depth = points_cam[..., :, 2] > z_min
    valid = finite & positive_depth

    safe_numerator = torch.where(
        valid[..., :, None],
        points_t[..., :, :2],
        torch.zeros_like(points_t[..., :, :2])
    )

    safe_depth = torch.where(
        valid[..., :, None],
        points_t[..., :, 2:3],
        torch.ones_like(points_t[..., :, 2:3])
    )

    uv = safe_numerator / safe_depth

    uv = torch.where(
        valid[..., :, None],
        uv,
        torch.full_like(uv, float("nan"))
    )

    return uv, valid

def project_weak(points_cam, K, z_ref, z_min=1e-4):
    points_ref = points_cam.clone()
    points_ref[..., 2] = z_ref
    points_t = torch.matmul(points_ref, K.transpose(-1, -2))
    finite = torch.isfinite(points_cam).all(dim=-1)
    positive_depth = points_cam[..., :, 2] > z_min
    valid = finite & positive_depth

    safe_numerator = torch.where(
        valid[..., :, None],
        points_t[..., :, :2],
        torch.zeros_like(points_t[..., :, :2])
    )

    depth = torch.full_like(
        safe_numerator, 
        z_ref
    )

    uv = safe_numerator / depth

    uv = torch.where(
        valid[..., :, None],
        uv,
        torch.full_like(uv, float("nan"))
    )

    return uv, valid

def apply_crop_affine(uv, A):
    ones = torch.ones(*uv.shape[:-1], 1, dtype=uv.dtype, device=uv.device)
    uv = torch.cat((uv, ones), dim=-1)
    
    uv_ = torch.matmul(uv, A.transpose(-1, -2))

    return uv_[..., :2]