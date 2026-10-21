
import math
from pathlib import Path

import torch

from src.motion import generate_elbow_motion
from src.smpl_runner import load_smpl_model, smpl_forward
from experiments.visualization import plot_smpl_motion


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/smpl_clean/SMPL_NEUTRAL.pkl"
RESULT_DIR = ROOT / "results/smpl/motion"


def exp1_motion_parameters():
    print("\n===== Exp 1: Motion Parameters =====")

    T = 60
    body_pose = generate_elbow_motion(num_frames=T)

    assert body_pose.shape == (T, 69)
    assert torch.isfinite(body_pose).all()

    angles = body_pose[:, 53]

    # Check that only the left elbow Z component changes.
    other_params = body_pose.clone()
    other_params[:, 53] = 0

    assert torch.allclose(
        other_params,
        torch.zeros_like(other_params),
    )

    # Verify the prescribed cosine trajectory.
    t = torch.linspace(0, 1, T)
    expected = math.pi / 4 * (
        1 - torch.cos(2 * math.pi * t)
    )

    assert torch.allclose(angles, expected, atol=1e-5)

    print("Body pose:", body_pose.shape)
    print("Start angle:", math.degrees(angles[0].item()))
    print("Max angle:", math.degrees(angles.max().item()))
    print("End angle:", math.degrees(angles[-1].item()))

    assert abs(angles[0].item()) < 1e-5
    assert abs(angles[-1].item()) < 1e-5

    print("PASS")
    return body_pose


@torch.no_grad()
def exp2_smpl_forward(model, body_pose):
    print("\n===== Exp 2: SMPL Motion Forward =====")

    T = body_pose.shape[0]
    device = model.v_template.device
    dtype = model.v_template.dtype

    body_pose = body_pose.to(device=device, dtype=dtype)

    betas = torch.zeros(T, model.num_betas, device=device, dtype=dtype)
    global_orient = torch.zeros(T, 3, device=device, dtype=dtype)
    transl = torch.zeros(T, 3, device=device, dtype=dtype)

    V, J = smpl_forward(
        model,
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
    )

    print("Vertices:", V.shape)
    print("Joints:", J.shape)

    assert V.shape == (T, 6890, 3)
    assert J.shape[0] == T and J.shape[-1] == 3
    assert J.shape[1] >= 24

    assert torch.isfinite(V).all()
    assert torch.isfinite(J).all()

    # First and last frames should show the same pose.
    endpoint_error = (V[0] - V[-1]).abs().max().item()

    # Root should stay fixed when global transforms are zero.
    root_error = (J[:, 0, :] - J[0, 0, :]).abs().max().item()

    print("Endpoint vertex error:", endpoint_error)
    print("Root joint variation:", root_error)

    assert endpoint_error < 1e-4
    assert root_error < 1e-4

    print("PASS")
    return V, J


@torch.no_grad()
def exp3_motion_continuity(V, J, faces):
    print("\n===== Exp 3: Motion Continuity =====")

    # SMPL left wrist joint index = 20
    wrist = J[:, 20, :]

    wrist_step = torch.linalg.vector_norm(
        wrist[1:] - wrist[:-1],
        dim=-1,
    )

    wrist_displacement = torch.linalg.vector_norm(
        wrist - wrist[0],
        dim=-1,
    )

    vertex_step = torch.linalg.vector_norm(
        V[1:] - V[:-1],
        dim=-1,
    )

    print(
        "Max wrist displacement:",
        wrist_displacement.max().item(),
    )
    print(
        "Max adjacent wrist step:",
        wrist_step.max().item(),
    )
    print(
        "Mean adjacent vertex step:",
        vertex_step.mean().item(),
    )
    print(
        "Max adjacent vertex step:",
        vertex_step.max().item(),
    )

    assert wrist_displacement.max().item() > 1e-3
    assert torch.isfinite(wrist_step).all()
    assert torch.isfinite(vertex_step).all()

    plot_smpl_motion(
        vertices=V,
        joints=J,
        faces=faces,
        save_dir=RESULT_DIR,
        fps=12,
    )

    print("PASS")


def run_all():
    model = load_smpl_model(MODEL_PATH, device="cpu")

    body_pose = exp1_motion_parameters()
    V, J = exp2_smpl_forward(model, body_pose)
    exp3_motion_continuity(V, J, model.faces)

    print("\n===== Day 6: ALL PASSED =====")


if __name__ == "__main__":
    run_all()
