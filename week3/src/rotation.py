import torch
import math

def skew(omega):
    if not torch.is_tensor(omega):
        omega = torch.tensor(omega, dtype=torch.get_default_dtype())
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
    if not torch.is_tensor(omega):
        omega = torch.tensor(omega, dtype=torch.get_default_dtype())

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

def matrix_to_rotation_6D(R):
    
    return R[..., :2, :].reshape(*R.shape[:-2], 6)

def rotation_6D_to_matrix(rot, eps=1e-8):
    if rot.shape[-1] != 6:
        raise ValueError("Incorrect rotation")


    a1 = rot[..., :3]
    a2 = rot[..., 3:]

    norm_1 = torch.linalg.vector_norm(a1, dim=-1)

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim = True) * b1

    norm_2 = torch.linalg.vector_norm(b2, dim=-1)

    valid = (norm_1 > eps) & (norm_2 > eps)

    if not valid.all():
        raise ValueError('Incorrect Input')

    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)

def quaternion_to_matrix(q, eps=1e-8):
    if q.shape[-1] != 4:
        raise ValueError(f"Expected [..., 4], got {q.shape}")

    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)

    if torch.any(norm < eps):
        raise ValueError("Zero or near-zero quaternion is invalid")

    q = q / norm

    w, x, y, z = q.unbind(dim=-1)

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)

    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)

    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    row0 = torch.stack((r00, r01, r02), dim=-1)
    row1 = torch.stack((r10, r11, r12), dim=-1)
    row2 = torch.stack((r20, r21, r22), dim=-1)

    return torch.stack((row0, row1, row2), dim=-2)



