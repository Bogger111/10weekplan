import torch
from multihmr2.api import init_hmr_session, infer_image


def perspective_projection(points, K):
    """
    points: [N, J, 3]
    K:      [3, 3]
    """
    points = points.float()
    K = K.float()

    uvw = points @ K.T

    valid = (
        torch.isfinite(points).all(dim=-1)
        & (points[..., 2] > 1e-6)
        & torch.isfinite(uvw).all(dim=-1)
    )

    pixels = torch.full_like(
        uvw[..., :2], float("nan")
    )

    pixels[valid] = (
        uvw[..., :2][valid]
        / uvw[..., 2:3][valid]
    )

    return pixels, valid


# ===== 1. 合成数据验证 =====

K_test = torch.tensor([
    [500., 0., 320.],
    [0., 500., 240.],
    [0., 0., 1.]
])

points_test = torch.tensor([[
    [1., 0., 5.],
    [1., 0., 10.]
]])

pixels_test, valid_test = perspective_projection(
    points_test, K_test
)

expected = torch.tensor([[
    [420., 240.],
    [370., 240.]
]])

assert valid_test.all()
assert torch.allclose(pixels_test, expected)

print("[PASS] Synthetic projection")


# ===== 2. 真实模型推理 =====

session = init_hmr_session(
    "/home/bogger/research-repos/multi-hmr2/checkpoints/multihmr2.pt"
)

pred = infer_image(
    session,
    "/home/bogger/research-repos/multi-hmr2/demo_data/sample_image.jpg",
    conf_thresh=0.4
)

K = pred.K
j3d = pred.persons.j3d
j2d = pred.persons.j2d

print("\n===== Real Prediction =====")
print("Persons:", len(pred))
print("K:", K.shape)
print("J3D:", j3d.shape)
print("J2D:", j2d.shape)


# ===== 3. 自己进行投影 =====

manual_j2d, valid = perspective_projection(
    j3d, K
)

official_j2d = j2d.float()

valid = valid & torch.isfinite(
    official_j2d
).all(dim=-1)


# ===== 4. 数值一致性检查 =====

print("\n===== Projection Consistency =====")

if not valid.any().item():
    print("No valid points; residual unavailable.")
else:
    residual = torch.linalg.vector_norm(
        manual_j2d[valid] - official_j2d[valid],
        dim=-1
    )

    print("Valid joints:", valid.sum().item())
    print("Mean error (px):", residual.mean().item())
    print("Max error (px):", residual.max().item())
    print("Median error (px):", residual.median().item())

    print("\nFirst 5 projected points:")
    print(manual_j2d[valid][:5])

    print("\nFirst 5 official points:")
    print(official_j2d[valid][:5])