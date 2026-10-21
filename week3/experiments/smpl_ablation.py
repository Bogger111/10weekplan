
import math
from pathlib import Path

import torch
from smplx.lbs import lbs

from src.smpl_runner import load_smpl_model, smpl_forward
from experiments.visualization import plot_smpl_pose_ablation


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/smpl_clean/SMPL_NEUTRAL.pkl"


def make_params(model):
    device = model.v_template.device
    dtype = model.v_template.dtype

    return {
        "betas": torch.zeros(1, model.num_betas, device=device, dtype=dtype),
        "body_pose": torch.zeros(1, 69, device=device, dtype=dtype),
        "global_orient": torch.zeros(1, 3, device=device, dtype=dtype),
        "transl": torch.zeros(1, 3, device=device, dtype=dtype),
    }


def forward_direct_lbs(model, params, posedirs):
    """Call official LBS with a selectable pose-corrective basis."""

    full_pose = torch.cat(
        [params["global_orient"], params["body_pose"]],
        dim=1,
    )

    return lbs(
        betas=params["betas"],
        pose=full_pose,
        v_template=model.v_template,
        shapedirs=model.shapedirs,
        posedirs=posedirs,
        J_regressor=model.J_regressor,
        parents=model.parents,
        lbs_weights=model.lbs_weights,
        pose2rot=True,
    )


@torch.no_grad()
def run_all():
    model = load_smpl_model(MODEL_PATH, device="cpu")
    zero_posedirs = torch.zeros_like(model.posedirs)

    print("\n===== Exp 6: Pose Blend Shapes Ablation =====")
    print("Pose basis:", tuple(model.posedirs.shape))

    # Control A: at zero pose, corrective offsets should vanish.
    params = make_params(model)

    V_rest_full, _ = smpl_forward(model, **params)
    V_rest_off, _ = forward_direct_lbs(
        model, params, zero_posedirs
    )

    rest_error = (
        V_rest_full - V_rest_off
    ).abs().max().item()

    print("Rest pose error:", rest_error)
    assert rest_error < 1e-5

    # Control B: bend left elbow by 90 degrees.
    elbow_joint = 18
    elbow_offset = (elbow_joint - 1) * 3
    params["body_pose"][0, elbow_offset + 2] = math.pi / 2

    # Full SMPL through our src wrapper
    V_full, J_full = smpl_forward(model, **params)

    # Validate that our direct official LBS call is equivalent.
    V_direct, J_direct = forward_direct_lbs(
        model, params, model.posedirs
    )

    implementation_error = (
        V_full - V_direct
    ).abs().max().item()

    print("Direct LBS vs SMPL error:", implementation_error)
    assert implementation_error < 1e-5

    # Only change: disable pose-corrective blend shapes.
    V_off, J_off = forward_direct_lbs(
        model, params, zero_posedirs
    )

    delta = torch.linalg.vector_norm(
        V_full[0] - V_off[0], dim=-1
    )

    core_joint_error = (
        J_direct - J_off
    ).abs().max().item()

    print("Mean vertex displacement:", delta.mean().item())
    print("Max vertex displacement:", delta.max().item())
    print("Core joint difference:", core_joint_error)

    # Inspect the region around the elbow.
    elbow_center = J_full[0, elbow_joint]

    distance_to_elbow = torch.linalg.vector_norm(
        V_full[0] - elbow_center, dim=-1
    )

    near_mask = distance_to_elbow < 0.25

    if near_mask.any():
        print(
            "Near-elbow mean displacement:",
            delta[near_mask].mean().item(),
        )

    assert torch.isfinite(V_off).all()
    assert delta.max().item() > 1e-5
    assert core_joint_error < 1e-5

    # Original model was never changed.
    V_repeat, _ = smpl_forward(model, **params)
    assert torch.allclose(V_repeat, V_full, atol=1e-6)

    plot_smpl_pose_ablation(
        vertices_full=V_full,
        vertices_off=V_off,
        joints_full=J_full,
        faces=model.faces,
        save_path=ROOT / "results/smpl/pose_ablation.png",
    )

    print("PASS")


if __name__ == "__main__":
    run_all()
