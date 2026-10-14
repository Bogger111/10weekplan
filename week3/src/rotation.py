import torch
import math

def skew(omega):
    omega = torch.as_tensor(
        omega,
        dtype=torch.float32
    )

    assert omega.shape[-1] == 3

    w1, w2, w3 = omega.unbind(-1)

    hat = torch.zeros(
        *omega.shape[:-1],
        3, 3,
        dtype=omega.dtype,
        device=omega.device
    )

    hat[..., 0, 1] = -w3
    hat[..., 0, 2] =  w2

    hat[..., 1, 0] =  w3
    hat[..., 1, 2] = -w1

    hat[..., 2, 0] = -w2
    hat[..., 2, 1] =  w1

    return hat

def axis_angle_to_matrix(omega):
    omega = torch.as_tensor(
        omega,
        dtype=torch.float32
    )

    assert omega.shape[-1] == 3

    theta = torch.linalg.norm(
        omega,
        dim=-1
    )

    K = skew(omega)

    I = torch.eye(
        3,
        dtype=omega.dtype,
        device=omega.device
    ).expand(
        *omega.shape[:-1],
        3, 3
    )

    A = torch.sinc(theta / torch.pi)

    B = 0.5 * torch.sinc(
        theta / (2 * torch.pi)
    ) ** 2

    R = (
        I
        + A[..., None, None] * K
        + B[..., None, None] * (K @ K)
    )

    return R

def apply_rotation(R, points):

    return torch.matmul(R, points.transpose(-1, -2)).transpose(-1, -2)

def check_so3(R, atol=1e-5):
    identity = torch.eye(
        3,
        dtype=R.dtype,
        device=R.device
    ).expand(*R.shape[:-2], 3, 3)

    RtR = R @ R.transpose(-1, -2)

    finite = torch.isfinite(R).all()

    orth_error = (RtR - identity).abs().max()

    det = torch.linalg.det(R)
    det_error = (det - 1).abs().max()

    orthogonal = orth_error < atol
    proper = det_error < atol

    return {
        "finite": finite.item(),
        "orthogonal": orthogonal.item(),
        "det": det,
        "det_is_one": proper.item(),
        "orth_error": orth_error.item(),
        "det_error": det_error.item(),
    }
