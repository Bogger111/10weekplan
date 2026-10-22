
import csv
from pathlib import Path

import numpy as np
import torch

from week4.src.camera import (
    to_camera,
    project_perspective,
)
from week4.src.metrics import (
    mpjpe,
    root_aligned_mpjpe,
    pa_mpjpe,
)

DTYPE = torch.float64
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "week4/results"


def load_day3():
    path = RESULTS / "d3_reprojection_inputs.npz"

    with np.load(path) as data:
        joints = torch.as_tensor(
            data["joints"].copy(), dtype=DTYPE
        ).reshape(1, 24, 3)

        R = torch.as_tensor(
            data["R"].copy(), dtype=DTYPE
        ).reshape(1, 3, 3)

        t = torch.as_tensor(
            data["t_gt"].copy(), dtype=DTYPE
        ).reshape(1, 3)

        K = torch.as_tensor(
            data["K"].copy(), dtype=DTYPE
        ).reshape(1, 3, 3)

        uv_gt = torch.as_tensor(
            data["uv_gt"].copy(), dtype=DTYPE
        ).reshape(1, 24, 2)

        valid_gt = torch.as_tensor(
            data["valid_gt"].copy(), dtype=torch.bool
        ).reshape(1, 24)

        saved_camera = torch.as_tensor(
            data["camera_points"].copy(), dtype=DTYPE
        ).reshape(1, 24, 3)

    return joints, R, t, K, uv_gt, valid_gt, saved_camera


def evaluate(name, predicted_3d, gt_3d, gt_uv, K):
    uv, valid = project_perspective(predicted_3d, K)

    assert valid.all(), f"{name}: invalid depth"

    pixel_error = torch.linalg.vector_norm(
        uv - gt_uv, dim=-1
    ).mean().item()

    raw = mpjpe(predicted_3d, gt_3d).item()

    root = root_aligned_mpjpe(
        predicted_3d, gt_3d
    ).item()

    pa = pa_mpjpe(predicted_3d, gt_3d).item()

    print(
        f"{name:18s} | "
        f"2D={pixel_error:.8f}px | "
        f"Raw={raw:.8f} | "
        f"Root={root:.8f} | "
        f"PA={pa:.8f}"
    )

    return {
        "case": name,
        "mean_2d_error_px": pixel_error,
        "raw_mpjpe": raw,
        "root_mpjpe": root,
        "pa_mpjpe": pa,
    }


def main():
    (
        joints, R, t, K,
        uv_saved, valid_saved, camera_saved
    ) = load_day3()

    print("\n===== D5 Exp 1: Reproduce Day 3 =====")

    gt_camera = to_camera(joints, R, t)
    uv_gt, valid = project_perspective(gt_camera, K)

    camera_diff = (
        gt_camera - camera_saved
    ).abs().max().item()

    pixel_diff = (
        uv_gt[valid] - uv_saved[valid]
    ).abs().max().item()

    print("Camera coordinate diff:", camera_diff)
    print("Saved 2D projection diff:", pixel_diff)

    assert torch.equal(valid, valid_saved)
    assert camera_diff < 1e-5
    assert pixel_diff < 1e-3

    print("\n===== D5 Exp 2: Controlled Errors =====")

    # A. Translation in camera coordinates
    delta = torch.tensor(
        [0.1, 0.0, 0.0], dtype=DTYPE
    ).reshape(1, 1, 3)

    case_a = gt_camera + delta

    # B. Global similarity scaling about camera origin
    case_b = 1.3 * gt_camera

    # C. Move one joint along its camera ray
    case_c = gt_camera.clone()
    case_c[:, 20, :] *= 1.3

    cases = {
        "baseline": gt_camera,
        "translation_x": case_a,
        "camera_scale": case_b,
        "single_joint_ray": case_c,
    }

    rows = [
        evaluate(name, pred, gt_camera, uv_gt, K)
        for name, pred in cases.items()
    ]

    print("\n===== D5 Exp 3: Assertions =====")

    baseline, a, b, c = rows

    assert baseline["mean_2d_error_px"] < 1e-8
    assert baseline["raw_mpjpe"] < 1e-8

    # A: 3D translation is visible in 2D,
    # but removable by root / PA alignment.
    assert a["mean_2d_error_px"] > 1.0
    assert abs(a["raw_mpjpe"] - 0.1) < 1e-6
    assert a["root_mpjpe"] < 1e-6
    assert a["pa_mpjpe"] < 1e-6

    # B: Global scaling preserves perspective UV.
    assert b["mean_2d_error_px"] < 1e-6
    assert b["raw_mpjpe"] > 0.1
    assert b["pa_mpjpe"] < 1e-6

    # C: Ray movement preserves one joint's UV
    # while changing its 3D structure.
    assert c["mean_2d_error_px"] < 1e-6
    assert c["raw_mpjpe"] > 1e-3
    assert c["pa_mpjpe"] > 1e-4

    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "d5_integration.csv"

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", output)
    print("D5 INTEGRATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
