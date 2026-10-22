
import math
import torch
import torch.nn.functional as F

from week3.src.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6D,
    rotation_6D_to_matrix,
)

DTYPE = torch.float64
ATOL = 1e-10


def hmr2_rot6d_reference(x: torch.Tensor):
    """
    HMR2 column-based 6D rotation convention.
    Reference: hmr2/utils/geometry.py
    Input: [N, 6]
    Output: [N, 3, 3]
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()

    a1 = x[:, :, 0]
    a2 = x[:, :, 1]

    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(
        a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1,
        dim=-1,
    )
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-1)


def hmr2_matrix_to_6d(R):
    """Encode the first two COLUMNS, preserving their order."""
    return R[..., :, :2].transpose(-1, -2).reshape(
        *R.shape[:-2], 6
    )


def check_close(actual, expected, name):
    err = (actual - expected).abs().max().item()
    print(f"{name}: max error = {err:.3e}")
    assert err < ATOL, name


def main():
    # Two known rotations
    aa = torch.tensor([
        [0.0, 0.0, math.pi / 3],
        [0.2, -0.3, 0.4],
    ], dtype=DTYPE)

    R_gt = axis_angle_to_matrix(aa)

    print("\n===== Exp 1: Week3 Roundtrip =====")
    x_week3 = matrix_to_rotation_6D(R_gt)
    R_week3 = rotation_6D_to_matrix(x_week3)

    check_close(R_week3, R_gt, "Week3 reconstruction")

    print("\n===== Exp 2: HMR2 Roundtrip =====")
    x_hmr2 = hmr2_matrix_to_6d(R_gt)
    R_hmr2 = hmr2_rot6d_reference(x_hmr2)

    check_close(R_hmr2, R_gt, "HMR2 reconstruction")

    print("\n===== Exp 3: Same 6D Input =====")
    R_wrong = hmr2_rot6d_reference(x_week3)

    direct_error = (R_wrong - R_week3).abs().max().item()
    print(f"Direct difference: {direct_error:.6f}")

    assert direct_error > 0.1

    check_close(
        R_wrong,
        R_week3.transpose(-1, -2),
        "Transpose relationship",
    )

    print("\n===== Exp 4: Random 6D =====")
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(
        128, 6, dtype=DTYPE, generator=generator
    )

    R_w = rotation_6D_to_matrix(x)
    R_h = hmr2_rot6d_reference(x)

    check_close(
        R_w.transpose(-1, -2),
        R_h,
        "Random transpose relationship",
    )

    I = torch.eye(3, dtype=DTYPE)

    check_close(
        R_w @ R_w.transpose(-1, -2),
        I,
        "Week3 orthogonality",
    )

    check_close(
        R_h @ R_h.transpose(-1, -2),
        I,
        "HMR2 orthogonality",
    )

    assert torch.allclose(
        torch.linalg.det(R_w),
        torch.ones(128, dtype=DTYPE),
        atol=ATOL,
    )
    assert torch.allclose(
        torch.linalg.det(R_h),
        torch.ones(128, dtype=DTYPE),
        atol=ATOL,
    )

    print("\nD5 Rotation 6D Convention: ALL PASSED")


if __name__ == "__main__":
    main()
