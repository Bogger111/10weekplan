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


print("\n===== Precision Check =====")

print("K dtype:", K.dtype)
print("J3D dtype:", j3d.dtype)
print("J2D dtype:", j2d.dtype)

# 模拟将手动投影量化到官方输出精度
quantized = manual_j2d.to(j2d.dtype).float()

quant_error = torch.linalg.vector_norm(
    quantized[valid] - official_j2d[valid],
    dim=-1
)

print("After dtype matching:")
print("Mean residual:", quant_error.mean().item())
print("Max residual:", quant_error.max().item())

print("\n===== Anny Mesh Placement =====")

persons = pred.persons

# 默认 Full-resolution Anny 的参考顶点
center = persons.v3d[:, 5054, :].float()

# 最终 Pelvis 位置
pelvis = persons.transl_pelvis.float()

# 最终 J3D 中的根关节
root = persons.j3d[:, 0, :].float()

print("Center shape:", center.shape)
print("Pelvis shape:", pelvis.shape)
print("Root shape:", root.shape)

assert center.shape == pelvis.shape == root.shape

# 检验 1：Pelvis 是否就是 J3D 的根关节
root_error = (pelvis - root).abs()

print("Root max error:", root_error.max().item())

# 检验 2：Mesh 参考点与 Pelvis 的距离
center_pelvis_dist = torch.linalg.vector_norm(
    center - pelvis,
    dim=-1
)

print("Center-Pelvis distances:")
print(center_pelvis_dist)

print("Mean distance:", center_pelvis_dist.mean().item())
print("Max distance:", center_pelvis_dist.max().item())

persons = pred.persons

pelvis = persons.transl_pelvis.float()
conf = persons.conf.flatten().float()

print("\n===== Camera-Space Layout =====")

for i in range(len(pred)):
    x, y, z = pelvis[i].tolist()
    print(
        f"Person {i:02d} | "
        f"conf={conf[i]:.3f} | "
        f"X={x:.3f} Y={y:.3f} Z={z:.3f}"
    )

# 先以第 0、1 个预测做数值示例
# 后续根据可视化确认它们的身份
i, j = 0, 1

delta = pelvis[j] - pelvis[i]
distance = torch.linalg.vector_norm(delta)

print("\n===== Relative Position =====")
print("Delta XYZ:", delta)
print("3D distance:", distance.item())
print("Depth difference:", delta[2].item())

import matplotlib.pyplot as plt

persons = pred.persons

xy = persons.j2d[:, 0, :].float().cpu().numpy()
xyz = persons.transl_pelvis.float().cpu().numpy()
conf = persons.conf.flatten().float().cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(xy[:, 0], xy[:, 1])

for i, ((u, v), p, c) in enumerate(zip(xy, xyz, conf)):
    ax.annotate(
        f"{i}: Z={p[2]:.1f}, c={c:.2f}",
        (u, v),
        xytext=(5, -6),
        textcoords="offset points",
        fontsize=8,
    )

ax.invert_yaxis()
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Image u")
ax.set_ylabel("Image v")
ax.set_title("Multi-HMR 2: Projected Pelvis Layout")

plt.tight_layout()
plt.savefig(
    "/home/bogger/10weekplan/week6/results/pelvis_layout.png",
    dpi=150
)
plt.close()



from PIL import Image, ImageDraw
from multihmr2.utils.image import (
    nonsquare_and_resize_image_without_K
)

img = Image.open("/home/bogger/research-repos/multi-hmr2/demo_data/sample_image.jpg").convert("RGB")

# 使用与官方推理相同的图像几何预处理
img = nonsquare_and_resize_image_without_K(
    img, 768, 16
)

draw = ImageDraw.Draw(img)

xy = pred.persons.j2d[:, 0, :].float().cpu()
depth = pred.persons.transl_pelvis[:, 2].float().cpu()

for i in range(len(pred)):
    u, v = xy[i].tolist()

    color = "red" if depth[i] < 20 else "yellow"

    draw.ellipse(
        (u - 5, v - 5, u + 5, v + 5),
        fill=color,
        outline="black"
    )

    draw.text(
        (u + 7, v - 12),
        str(i),
        fill=color,
        stroke_width=1,
        stroke_fill="black"
    )

img.save(
    "/home/bogger/10weekplan/week6/results/pelvis_overlay.png"
)