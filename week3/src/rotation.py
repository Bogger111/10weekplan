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
    omega = torch.as_tensor(omega, dtype=torch.float32)
    w1, w2, w3 = omega.unbind(-1)
    theta = torch.linalg.norm(omega, dim=-1)
    hat = skew(omega)

    identity = torch.zeros_like(hat, dtype=hat.dtype, device=hat.device)
    identity[..., 0, 0] = 1
    identity[..., 1, 1] = 1
    identity[..., 2, 2] = 1

    if theta == 0:
        matrix = identity
    
    else: matrix = identity + math.sin(theta) / theta * hat + (1 - math.cos(theta)) / theta ** 2 * (hat @ hat)

    return matrix

def apply_rotation(R, points):

    return torch.matmul(R, points.transpose(-1, -2)).transpose(-1, -2)

def check_so3(R):
    identity = torch.zeros_like(R, dtype=R.dtype, device=R.device)
    identity[..., 0, 0] = 1
    identity[..., 1, 1] = 1
    identity[..., 2, 2] = 1

    det = torch.linalg.det(R)
    
    finite = torch.isfinite(R).all() 
    o = torch.allclose(R @ R.transpose(-1, -2), identity)
    d = torch.allclose(det, torch.ones_like(det))

    print('R:', R)
    print(f"isfinite:{finite}")
    print(f"orthogonality:{o}")
    print(f"determinant:{d}")

zeros = torch.zeros(2,3,3)

identity = zeros.clone()
identity[..., 0, 0] = 1
identity[..., 1, 1] = 1
identity[..., 2, 2] = 1

axis_angle = [0.3, -0.2, 0.4]
x = [1., 0., 0.]
y = [0., 1., 0.]
z = [0., 0., 1.]

X = axis_angle_to_matrix(x)
Y = axis_angle_to_matrix(y)
Z = axis_angle_to_matrix(z)

TEST = axis_angle_to_matrix(axis_angle)

check_so3(zeros)
check_so3(identity)
check_so3(X)
check_so3(Y)
check_so3(Z)
check_so3(TEST)
    

R = TEST
RtR = R @ R.transpose(-1, -2)

print(RtR)
print((RtR - torch.eye(3)).abs().max())