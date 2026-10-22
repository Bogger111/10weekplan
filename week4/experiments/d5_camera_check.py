
from pathlib import Path

import numpy as np
import torch

from week4.src.camera import (
    to_camera,
    project_perspective,
)

DTYPE = torch.float64
ROOT = Path(__file__).resolve().parents[2]


def check(actual, expected, label, atol=1e-10):
    err = (actual - expected).abs().max().item()
    print(f"{label}: max error = {err:.3e}")
    assert err < atol, label


def main():
    # Use the saved SMPL 24 joints from Day 3
    path = ROOT / "week4/results/d3_reprojection_inputs.npz"

    with np.load(path) as data:
        joints = torch.as_tensor(
            data["joints"].copy(), dtype=DTYPE
        ).reshape(1, 24, 3)

    # Two camera predictions, same human geometry
    points = joints.expand(2, -1, -1).clone()
    B = points.shape[0]

    pred_cam = torch.tensor([
        [1.25,  0.10, -0.05],
        [0.90, -0.06,  0.08],
    ], dtype=DTYPE)

    f = 5000.0
    S = 256.0

    print("\n===== Exp 1: Camera Translation =====")

    tz = 2 * f / (S * pred_cam[:, 0])

    cam_t = torch.stack([
        pred_cam[:, 1],
        pred_cam[:, 2],
        tz,
    ], dim=-1)

    print("pred_cam_t:", cam_t)

    expected_z = 2 * f / (S * pred_cam[:, 0])
    check(cam_t[:, 2], expected_z, "Depth formula")

    print("\n===== Exp 2: HMR2 Projection =====")

    R = torch.eye(3, dtype=DTYPE).expand(B, -1, -1)
    points_cam = to_camera(points, R, cam_t)

    # HMR2 uses normalized focal length f / S
    K_norm = torch.eye(3, dtype=DTYPE).repeat(B, 1, 1)
    K_norm[:, 0, 0] = f / S
    K_norm[:, 1, 1] = f / S
    K_norm[:, 0, 2] = 0
    K_norm[:, 1, 2] = 0

    uv_norm, valid = project_perspective(
        points_cam, K_norm
    )
    assert valid.all()

    # Independent coordinate-wise formula
    reference = (
        (points[..., :2] + cam_t[:, None, :2])
        / (points[..., 2:3] + cam_t[:, None, 2:3])
        * (f / S)
    )

    check(uv_norm, reference, "HMR2 normalized projection")

    print("\n===== Exp 3: Pixel Coordinate Conversion =====")

    # Equivalent projection into a centered S x S crop
    K_pixel = torch.eye(3, dtype=DTYPE).repeat(B, 1, 1)
    K_pixel[:, 0, 0] = f
    K_pixel[:, 1, 1] = f
    K_pixel[:, 0, 2] = S / 2
    K_pixel[:, 1, 2] = S / 2

    uv_pixel, valid_pixel = project_perspective(
        points_cam, K_pixel
    )
    assert valid_pixel.all()

    reconstructed_pixel = uv_norm * S + S / 2

    check(
        reconstructed_pixel,
        uv_pixel,
        "Centered crop pixel conversion",
    )

    print("\n===== Exp 4: Avoid Double Translation =====")

    # Incorrect example: applying cam_t a second time
    points_double = to_camera(points_cam, R, cam_t)
    uv_double, valid_double = project_perspective(
        points_double, K_norm
    )
    assert valid_double.all()

    double_error = (uv_double - uv_norm).abs().max().item()
    print(f"Double translation changes UV by {double_error:.6f}")

    assert double_error > 1e-4

    print("\nD5 Camera Interface: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
