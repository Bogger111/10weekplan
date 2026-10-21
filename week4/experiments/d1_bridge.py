
import json
from pathlib import Path

import numpy as np
import torch

from week3.src.smpl_runner import load_smpl_model, smpl_forward


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "week04/results"
MODEL_PATH = ROOT / "week3/models/smpl_clean/SMPL_NEUTRAL.pkl"


def exp1_tensor_shapes():
    print("\n===== Exp 1: Spatio-Temporal Shapes =====")

    B, T, J, C = 2, 9, 17, 32

    x = torch.arange(
        B * T * J * C,
        dtype=torch.float64,
    ).reshape(B, T, J, C)

    # Spatial: each frame attends over joints.
    spatial = x.reshape(B * T, J, C)
    spatial_back = spatial.reshape(B, T, J, C)

    # Temporal per joint: each joint attends over time.
    temporal_joint = x.permute(0, 2, 1, 3).reshape(
        B * J, T, C
    )
    temporal_back = temporal_joint.reshape(
        B, J, T, C
    ).permute(0, 2, 1, 3)

    # Temporal per frame: the entire frame is one token.
    temporal_frame = x.reshape(B, T, J * C)
    frame_back = temporal_frame.reshape(B, T, J, C)

    assert torch.equal(x, spatial_back)
    assert torch.equal(x, temporal_back)
    assert torch.equal(x, frame_back)

    print("Original:", x.shape)
    print("Spatial:", spatial.shape)
    print("Temporal per joint:", temporal_joint.shape)
    print("Temporal per frame:", temporal_frame.shape)

    print("Spatial attention map:", (B * T, J, J))
    print("Joint temporal attention map:", (B * J, T, T))
    print("Frame temporal attention map:", (B, T, T))

    # Missing observations need a separate visibility mask.
    visible = torch.ones(B, T, J, dtype=torch.bool)
    visible[0, 3, 5] = False

    assert visible.sum().item() == B * T * J - 1

    print("Shape round-trips: PASS")


@torch.no_grad()
def exp2_smpl_reference():
    print("\n===== Exp 2: SMPL Reference =====")

    model = load_smpl_model(MODEL_PATH, device="cpu")

    dtype = model.v_template.dtype

    params = {
        "betas": torch.zeros(1, 10, dtype=dtype),
        "body_pose": torch.zeros(1, 69, dtype=dtype),
        "global_orient": torch.zeros(1, 3, dtype=dtype),
        "transl": torch.zeros(1, 3, dtype=dtype),
    }

    V, J = smpl_forward(model, **params)

    assert V.shape == (1, 6890, 3)
    assert J.shape[0] == 1
    assert J.shape[-1] == 3

    # Controlled translation.
    shifted = {
        k: v.clone()
        for k, v in params.items()
    }
    shifted["transl"][0, 0] = 0.1

    V_shift, J_shift = smpl_forward(model, **shifted)

    expected = torch.tensor([0.1, 0.0, 0.0], dtype=dtype)

    v_error = ((V_shift - V) - expected).abs().max().item()
    j_error = ((J_shift - J) - expected).abs().max().item()

    print("Vertices:", V.shape)
    print("Joints:", J.shape)
    print("Translation vertex error:", v_error)
    print("Translation joint error:", j_error)

    assert v_error < 1e-5
    assert j_error < 1e-5

    RESULTS.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        RESULTS / "body_reference.npz",
        vertices=V[0].cpu().numpy(),
        joints=J[0].cpu().numpy(),
        faces=np.asarray(model.faces),
        **{k: v.cpu().numpy() for k, v in params.items()},
    )

    metadata = {
        "model_type": "smpl",
        "gender": "neutral",
        "num_betas": 10,
        "vertices_shape": list(V.shape),
        "joints_shape": list(J.shape),
        "frame": "SMPL model output frame; no camera transform",
        "unit": "model units; verify scale before metric reporting",
        "model_transl": [0.0, 0.0, 0.0],
        "camera_transform_applied": False,
        "joint_subset": "Not selected yet; verify convention",
    }

    (RESULTS / "body_reference_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Saved body_reference.npz")
    print("Saved body_reference_meta.json")
    print("PASS")


def run_all():
    exp1_tensor_shapes()
    exp2_smpl_reference()
    print("\n===== Week 04 Day 1: TESTS PASSED =====")


if __name__ == "__main__":
    run_all()
