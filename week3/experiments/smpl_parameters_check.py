
import math
from pathlib import Path

import torch

from src.smpl_runner import (
    load_smpl_model,
    smpl_forward,
)

from experiments.visualization import (
    plot_smpl_mesh_cases,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "smpl_clean"
    / "SMPL_NEUTRAL.pkl"
)


# ============================================================
# Helpers
# ============================================================

def make_params(model, B=1):
    device = model.v_template.device
    dtype = model.v_template.dtype

    return {
        "betas": torch.zeros(B, 10, device=device, dtype=dtype),
        "body_pose": torch.zeros(B, 69, device=device, dtype=dtype),
        "global_orient": torch.zeros(B, 3, device=device, dtype=dtype),
        "transl": torch.zeros(B, 3, device=device, dtype=dtype),
    }


def forward(model, params):
    return smpl_forward(model, **params)


def check_finite(*tensors):
    for tensor in tensors:
        assert torch.isfinite(tensor).all()


# ============================================================
# Exp 1
# Default SMPL
# ============================================================

def exp1_default_pose(model):

    params = make_params(model)

    V, J = forward(model, params)

    print("\n===== Exp 1: Default SMPL =====")
    print("Vertices:", V.shape)
    print("Joints:", J.shape)
    print("Faces:", model.faces.shape)

    assert V.shape == (1, 6890, 3)
    assert J.shape[0] == 1
    assert J.shape[-1] == 3

    check_finite(V, J)

    plot_smpl_mesh_cases(
        V,
        model.faces,
        ["Default SMPL"],
        "results/smpl/default_pose.png",
    )

    print("PASS")


# ============================================================
# Exp 2
# Shape parameters
# ============================================================

def exp2_shape(model):

    params = make_params(model, B=3)

    # Baseline: beta[0] = 0
    # Positive: beta[0] = +2
    # Negative: beta[0] = -2

    params["betas"][1, 0] = 2.0
    params["betas"][2, 0] = -2.0

    V, J = forward(model, params)

    delta_V = torch.linalg.vector_norm(
        V[1:] - V[0:1],
        dim=-1,
    )

    delta_J = torch.linalg.vector_norm(
        J[1:] - J[0:1],
        dim=-1,
    )

    print("\n===== Exp 2: Shape =====")

    print("Vertices:", V.shape)
    print("Mean vertex displacement:", delta_V.mean(dim=1))
    print("Max vertex displacement:", delta_V.max(dim=1).values)
    print("Mean joint displacement:", delta_J.mean(dim=1))

    assert V.shape == (3, 6890, 3)
    assert delta_V.max() > 1e-5
    check_finite(V, J)

    plot_smpl_mesh_cases(
        V,
        model.faces,
        ["Beta 0 = 0", "Beta 0 = +2", "Beta 0 = -2"],
        "results/smpl/shape_effects.png",
    )

    print("PASS")


# ============================================================
# Exp 3
# Local body pose
# ============================================================

def exp3_body_pose(model):

    params = make_params(model, B=2)

    # Standard SMPL joint index:
    # left elbow = 18
    # body_pose excludes joint 0 (pelvis)

    elbow_joint = 18
    elbow_index = (elbow_joint - 1) * 3

    # Axis-angle: rotate 60 degrees around local z
    params["body_pose"][1, elbow_index + 2] = math.pi / 3

    V, J = forward(model, params)

    vertex_difference = torch.linalg.vector_norm(
        V[1] - V[0],
        dim=-1,
    )

    elbow_difference = torch.linalg.vector_norm(
        J[1, 18] - J[0, 18]
    )

    wrist_difference = torch.linalg.vector_norm(
        J[1, 20] - J[0, 20]
    )

    print("\n===== Exp 3: Body Pose =====")

    print("Mean vertex displacement:", vertex_difference.mean())
    print("Elbow position change:", elbow_difference)
    print("Wrist position change:", wrist_difference)

    assert J.shape[1] >= 24
    assert vertex_difference.max() > 1e-5
    assert elbow_difference < 5e-4
    assert wrist_difference > 1e-3
    check_finite(V, J)

    plot_smpl_mesh_cases(
        V,
        model.faces,
        ["Rest Pose", "Left Elbow Rz(60 deg)"],
        "results/smpl/pose_effects.png",
    )

    print("PASS")


# ============================================================
# Exp 4
# Global orientation / translation
# ============================================================

def exp4_global_transform(model):

    params = make_params(model, B=3)

    # Case 0: Baseline

    # Case 1: Global rotation
    params["global_orient"][1, 2] = math.pi / 4

    # Case 2: Global translation
    translation = torch.tensor(
        [0.5, 0.0, 0.25],
        device=model.v_template.device,
        dtype=model.v_template.dtype,
    )

    params["transl"][2] = translation

    V, J = forward(model, params)

    translation_error = (
        (V[2] - V[0]) - translation
    ).abs().max()

    joint_translation_error = (
        (J[2] - J[0]) - translation
    ).abs().max()

    # Rigid rotation should preserve distances
    # from the root joint to the vertices.
    root_distance_before = torch.linalg.vector_norm(
        V[0] - J[0, 0],
        dim=-1,
    )

    root_distance_after = torch.linalg.vector_norm(
        V[1] - J[1, 0],
        dim=-1,
    )

    rotation_distance_error = (
        root_distance_before - root_distance_after
    ).abs().max()

    print("\n===== Exp 4: Global Transform =====")

    print("Translation error:", translation_error)
    print("Joint translation error:", joint_translation_error)
    print("Rotation distance error:", rotation_distance_error)

    assert translation_error < 1e-4
    assert joint_translation_error < 1e-4
    assert rotation_distance_error < 1e-4

    plot_smpl_mesh_cases(
        V,
        model.faces,
        ["Baseline", "Global Rz(45 deg)", "Translation"],
        "results/smpl/global_effects.png",
    )

    print("PASS")


# ============================================================
# Exp 5
# Batch consistency
# ============================================================

def exp5_batch(model):

    torch.manual_seed(42)

    B = 3
    params = make_params(model, B=B)

    params["betas"] = torch.randn_like(
        params["betas"]
    ) * 0.5

    params["body_pose"] = torch.randn_like(
        params["body_pose"]
    ) * 0.05

    params["transl"] = torch.randn_like(
        params["transl"]
    ) * 0.1

    V_batch, J_batch = forward(model, params)

    max_v_error = 0.0
    max_j_error = 0.0

    for b in range(B):

        single_params = {
            k: v[b:b + 1]
            for k, v in params.items()
        }

        V_single, J_single = forward(
            model,
            single_params,
        )

        max_v_error = max(
            max_v_error,
            (V_batch[b:b + 1] - V_single).abs().max().item(),
        )

        max_j_error = max(
            max_j_error,
            (J_batch[b:b + 1] - J_single).abs().max().item(),
        )

    print("\n===== Exp 5: Batch =====")

    print("Vertices:", V_batch.shape)
    print("Joints:", J_batch.shape)
    print("Vertex consistency error:", max_v_error)
    print("Joint consistency error:", max_j_error)

    assert V_batch.shape == (B, 6890, 3)
    assert max_v_error < 1e-4
    assert max_j_error < 1e-4

    print("PASS")


# ============================================================
# Run all
# ============================================================

def run_all():

    model = load_smpl_model(
        MODEL_PATH,
        device="cpu",
    )

    with torch.no_grad():
        exp1_default_pose(model)
        exp2_shape(model)
        exp3_body_pose(model)
        exp4_global_transform(model)
        exp5_batch(model)


if __name__ == "__main__":
    run_all()
