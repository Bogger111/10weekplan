
import csv
from pathlib import Path

import numpy as np
import torch

from week3.src.smpl_runner import (
    load_smpl_model,
    smpl_forward,
)
from week4.src.metrics import (
    mpjpe,
    root_aligned_mpjpe,
    pa_mpjpe,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    ROOT / "week3/models/smpl_clean/SMPL_NEUTRAL.pkl"
)
RESULT_DIR = ROOT / "week4/results"

DTYPE = torch.float64
ATOL = 1e-6


def check_close(x, expected, name, tol=ATOL):
    assert abs(x - expected) < tol, (
        f"{name}: got {x}, expected {expected}"
    )


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_smpl_model(
        MODEL_PATH, device="cpu"
    ).to(dtype=DTYPE).eval()

    betas = torch.zeros(1, 10, dtype=DTYPE)
    body_pose = torch.zeros(1, 69, dtype=DTYPE)
    orient = torch.zeros(1, 3, dtype=DTYPE)
    transl = torch.zeros(1, 3, dtype=DTYPE)

    @torch.no_grad()
    def forward(body_pose, orient, transl):
        _, joints = smpl_forward(
            model, betas, body_pose, orient, transl
        )
        return joints[:, :24, :].clone()

    # Ground truth: neutral SMPL body
    gt = forward(body_pose, orient, transl)

    print("GT shape:", tuple(gt.shape))
    assert gt.shape == (1, 24, 3)

    # Compare against the saved Day 3 reference.
    d3_path = RESULT_DIR / "d3_reprojection_inputs.npz"

    if d3_path.exists():
        with np.load(d3_path) as data:
            d3_joints = torch.as_tensor(
                data["joints"], dtype=DTYPE
            ).reshape(24, 3)

        max_diff = (gt[0] - d3_joints).abs().max().item()
        print("Day 3 reference max diff:", max_diff)
        check_close(max_diff, 0.0, "Day3 reference", 1e-5)
    else:
        print("Day 3 reference not found; skipping cross-check")

    # 1. Translation +0.1 along X
    t_new = transl.clone()
    t_new[0, 0] = 0.1
    pred_translation = forward(body_pose, orient, t_new)

    # 2. Global orientation around Y
    o_new = orient.clone()
    o_new[0, 1] = 0.35
    pred_rotation = forward(body_pose, o_new, transl)

    # 3. Global scale: geometric control, not SMPL beta
    pred_scale = gt * 1.3

    # 4. Local left elbow rotation
    pose_new = body_pose.clone()
    pose_new[0, 53] = 0.75
    pred_elbow = forward(pose_new, orient, transl)

    cases = {
        "identity": gt.clone(),
        "translation": pred_translation,
        "global_orient": pred_rotation,
        "scale_1.3": pred_scale,
        "left_elbow": pred_elbow,
    }

    rows = []

    for name, pred in cases.items():
        raw = mpjpe(pred, gt).item()
        root = root_aligned_mpjpe(pred, gt).item()
        pa = pa_mpjpe(pred, gt).item()

        wrist_delta = torch.linalg.vector_norm(
            pred[:, 20, :] - gt[:, 20, :], dim=-1
        ).mean().item()

        print(
            f"{name:15s} | "
            f"Raw={raw:.8f} | "
            f"Root={root:.8f} | "
            f"PA={pa:.8f}"
        )

        rows.append({
            "case": name,
            "raw_mpjpe": raw,
            "root_mpjpe": root,
            "pa_mpjpe": pa,
            "left_wrist_delta": wrist_delta,
        })

        if name == "identity":
            check_close(raw, 0.0, "Identity Raw")
            check_close(root, 0.0, "Identity Root")
            check_close(pa, 0.0, "Identity PA")

        elif name == "translation":
            check_close(raw, 0.1, "Translation Raw")
            check_close(root, 0.0, "Translation Root")
            check_close(pa, 0.0, "Translation PA")

        elif name == "global_orient":
            assert raw > 1e-3
            assert root > 1e-3
            check_close(pa, 0.0, "Global Orient PA")

        elif name == "scale_1.3":
            assert raw > 1e-3
            assert root > 1e-3
            check_close(pa, 0.0, "Scale PA")

        elif name == "left_elbow":
            assert raw > 1e-3
            assert root > 1e-3
            assert pa > 1e-4
            assert wrist_delta > 0.1

    output_path = RESULT_DIR / "d4_smpl_metrics.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", output_path)
    print("D4 Exp 14 SMPL Metrics Bridge: PASS")


if __name__ == "__main__":
    main()
