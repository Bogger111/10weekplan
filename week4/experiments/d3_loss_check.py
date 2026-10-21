
import csv
import json
from pathlib import Path

import numpy as np
import torch

from week4.src.camera import to_camera, project_perspective
from week4.src.reprojection import reprojection_loss
from week4.experiments.visualization import plot_d3_reprojection
from week3.src.smpl_runner import load_smpl_model, smpl_forward
from week4.experiments.visualization import plot_d3_parameter_controls


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DTYPE = torch.float64


def make_K():
    return torch.tensor([
        [800., 0., 320.],
        [0., 800., 240.],
        [0., 0., 1.],
    ], dtype=DTYPE)


# ============================================================
# D3 Exp 1: SMPL -> Synthetic 2D Ground Truth
# ============================================================

def exp1_smpl_projection():
    print("\n===== D3 Exp 1: SMPL -> 2D GT =====")

    with np.load(RESULTS / "body_reference.npz") as f:
        joints_np = f["joints"][:24].copy()
        vertices_np = f["vertices"].copy()

    joints = torch.tensor(joints_np, dtype=DTYPE)

    # Proper rotation: 180 degrees around camera X.
    R = torch.diag(
        torch.tensor([1., -1., -1.], dtype=DTYPE)
    )
    assert torch.det(R).item() > 0.999999

    K = make_K()
    t_gt = torch.tensor([0.10, -0.05, 3.0], dtype=DTYPE)

    cam_gt = to_camera(joints, R, t_gt)
    uv_gt, valid_gt = project_perspective(cam_gt, K)

    assert joints.shape == (24, 3)
    assert uv_gt.shape == (24, 2)
    assert valid_gt.all()
    assert torch.isfinite(uv_gt).all()

    # Independent scalar reference.
    uv_ref = torch.stack([
        800. * cam_gt[:, 0] / cam_gt[:, 2] + 320.,
        800. * cam_gt[:, 1] / cam_gt[:, 2] + 240.,
    ], dim=-1)

    torch.testing.assert_close(
        uv_gt, uv_ref, atol=1e-10, rtol=0
    )

    np.savez_compressed(
        RESULTS / "d3_reprojection_inputs.npz",
        joints=joints.numpy(),
        vertices=vertices_np,
        camera_points=cam_gt.numpy(),
        uv_gt=uv_gt.numpy(),
        valid_gt=valid_gt.numpy(),
        R=R.numpy(),
        t_gt=t_gt.numpy(),
        K=K.numpy(),
        joint_indices=np.arange(24),
    )

    config = {
        "source": "Week3 SMPL neutral reference",
        "gt_type": "synthetic",
        "joint_subset": "first 24 SMPL joints",
        "image_size": [640, 480],
        "initial_tx_ty": [0.0, 0.0],
        "gt_tx_ty": [0.10, -0.05],
        "fixed_tz": 3.0,
        "focal_length_px": [800, 800],
        "optimizer": "Adam",
        "learning_rate": 0.03,
        "steps": 200,
    }

    (RESULTS / "d3_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("SMPL joints:", joints.shape)
    print("2D joints:", uv_gt.shape)
    print("Valid:", valid_gt.sum().item())
    print("GT camera translation:", t_gt.tolist())
    print("PASS")

    return joints, R, K, t_gt, cam_gt, uv_gt, valid_gt


# ============================================================
# D3 Exp 2: Loss / Mask / Gradient
# ============================================================

def exp2_loss_contract():
    print("\n===== D3 Exp 2: Loss Contract =====")

    pred = torch.tensor([
        [420., 270.],
        [float("nan"), float("nan")],
    ], dtype=DTYPE, requires_grad=True)

    gt = torch.tensor([
        [400., 280.],
        [320., 240.],
    ], dtype=DTYPE)

    valid = torch.tensor([True, False])

    loss = reprojection_loss(
        pred, gt, valid, image_size=(640, 480)
    )

    expected = (20. / 640.) ** 2 + (-10. / 480.) ** 2

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    torch.testing.assert_close(
        loss, torch.tensor(expected, dtype=DTYPE),
        atol=1e-12, rtol=0
    )

    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.all(pred.grad[1] == 0)

    # No valid points must not silently yield zero.
    try:
        reprojection_loss(
            pred, gt,
            torch.zeros(2, dtype=torch.bool),
            image_size=(640, 480),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for zero valid joints"
        )

    print("Loss:", loss.item())
    print("Gradient:\n", pred.grad)
    print("Mask and gradient: PASS")


# ============================================================
# D3 Exp 3: Camera-only tx/ty Optimization
# ============================================================

def exp3_camera_fitting(data):
    print("\n===== D3 Exp 3: Camera Fitting =====")

    joints, R, K, t_gt, _, uv_gt, valid_gt = data

    theta = torch.nn.Parameter(
        torch.zeros(2, dtype=DTYPE)
    )
    optimizer = torch.optim.Adam([theta], lr=0.03)

    def forward():
        t = torch.cat((theta, t_gt[2:3]))
        points_cam = to_camera(joints, R, t)
        return project_perspective(points_cam, K)

    with torch.no_grad():
        uv_initial, _ = forward()

    initial_px = (
        (uv_initial[valid_gt] - uv_gt[valid_gt])
        .norm(dim=-1).mean().item()
    )

    history = []

    for step in range(201):
        optimizer.zero_grad()

        uv_pred, valid_pred = forward()
        valid = valid_gt & valid_pred

        loss = reprojection_loss(
            uv_pred, uv_gt, valid, image_size=(640, 480)
        )

        mean_px = (
            (uv_pred[valid] - uv_gt[valid])
            .norm(dim=-1).mean()
        )

        history.append([
            step,
            loss.item(),
            mean_px.item(),
            theta[0].item(),
            theta[1].item(),
        ])

        if step < 200:
            loss.backward()

            assert theta.grad is not None
            assert torch.isfinite(theta.grad).all()

            optimizer.step()

    with torch.no_grad():
        uv_final, valid_final = forward()

    final_px = (
        (uv_final[valid_gt] - uv_gt[valid_gt])
        .norm(dim=-1).mean().item()
    )

    # Independent inverse-projection check.
    base_cam = to_camera(
        joints, R,
        torch.tensor([0., 0., 3.], dtype=DTYPE)
    )

    z = base_cam[valid_gt, 2]

    tx_ref = (
        (uv_gt[valid_gt, 0] - 320.) * z / 800.
        - base_cam[valid_gt, 0]
    ).mean()

    ty_ref = (
        (uv_gt[valid_gt, 1] - 240.) * z / 800.
        - base_cam[valid_gt, 1]
    ).mean()

    ref = torch.stack([tx_ref, ty_ref])

    print("Initial mean pixel error:", initial_px)
    print("Final mean pixel error:", final_px)
    print("Recovered tx,ty:", theta.detach().tolist())
    print("GT tx,ty:", t_gt[:2].tolist())
    print("Analytical reference:", ref.tolist())

    assert torch.equal(valid_final, valid_gt)
    assert final_px < 0.1
    assert final_px < initial_px

    torch.testing.assert_close(
        theta.detach(), ref, atol=1e-3, rtol=0
    )

    with (RESULTS / "d3_fit.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "normalized_loss",
            "mean_pixel_error", "tx", "ty"
        ])
        writer.writerows(history)

    plot_d3_reprojection(
        uv_gt.detach().numpy(),
        uv_initial.detach().numpy(),
        uv_final.detach().numpy(),
        np.asarray(history),
        RESULTS / "reprojection_overlay.png",
    )

    print("PASS")


# ============================================================
# D3 Exp 4: Scale-Depth Ambiguity
# ============================================================

def exp4_scale_depth_ambiguity(data):
    print("\n===== D3 Exp 4: Scale-Depth Ambiguity =====")

    _, _, K, _, cam_gt, uv_gt, valid_gt = data

    rows = []

    for scale in [0.7, 1.0, 1.3]:
        scaled_cam = cam_gt * scale

        uv, valid = project_perspective(
            scaled_cam, K
        )

        assert torch.equal(valid, valid_gt)

        mean_px = (
            (uv[valid] - uv_gt[valid])
            .norm(dim=-1).mean().item()
        )

        mean_3d_change = (
            (scaled_cam - cam_gt)
            .norm(dim=-1).mean().item()
        )

        assert mean_px < 1e-8

        print(
            f"scale={scale:.1f}: "
            f"2D error={mean_px:.3e}px, "
            f"3D change={mean_3d_change:.4f}"
        )

        rows.append([
            "global_scale", scale,
            mean_px, mean_3d_change
        ])

    # Change only one point along its viewing ray.
    ray_changed = cam_gt.clone()
    ray_changed[5] *= 1.5

    uv_ray, valid_ray = project_perspective(
        ray_changed, K
    )

    assert torch.equal(valid_ray, valid_gt)

    ray_2d_error = (
        (uv_ray[valid_ray] - uv_gt[valid_ray])
        .norm(dim=-1).max().item()
    )

    ray_3d_change = (
        ray_changed[5] - cam_gt[5]
    ).norm().item()

    assert ray_2d_error < 1e-8
    assert ray_3d_change > 0

    print("Ray-change 2D max error:", ray_2d_error)
    print("Ray-change 3D displacement:", ray_3d_change)

    rows.append([
        "single_ray", 1.5,
        ray_2d_error, ray_3d_change
    ])

    with (RESULTS / "d3_controls.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "control", "scale",
            "pixel_error", "mean_3d_change"
        ])
        writer.writerows(rows)

    print("PASS")


# ============================================================
# D3 Exp 5: SMPL Parameter Controls
# ============================================================

@torch.no_grad()
def exp5_body_parameter_controls(data):
    print("\n===== D3 Exp 5: SMPL Parameter Controls =====")

    _, R, K, t_gt, _, _, _ = data

    model_path = (
        ROOT.parent
        / "week3/models/smpl_clean/SMPL_NEUTRAL.pkl"
    )
    model = load_smpl_model(model_path, device="cpu")
    dtype = model.v_template.dtype

    base_params = {
        "betas": torch.zeros(1, 10, dtype=dtype),
        "body_pose": torch.zeros(1, 69, dtype=dtype),
        "global_orient": torch.zeros(1, 3, dtype=dtype),
        "transl": torch.zeros(1, 3, dtype=dtype),
    }

    def make_case(params):
        _, joints = smpl_forward(model, **params)
        joints = joints[0, :24].to(DTYPE)

        camera_points = to_camera(joints, R, t_gt)
        uv, valid = project_perspective(camera_points, K)

        assert valid.all()
        return joints, uv

    J0, uv0 = make_case(base_params)

    # A. SMPL translation
    p = {k: v.clone() for k, v in base_params.items()}
    p["transl"][0, 0] = 0.1
    J_transl, uv_transl = make_case(p)

    expected_delta = torch.tensor(
        [0.1, 0., 0.], dtype=DTYPE
    )

    transl_error = (
        (J_transl - J0) - expected_delta
    ).abs().max().item()

    assert transl_error < 1e-5

    # B. Global orientation
    p = {k: v.clone() for k, v in base_params.items()}
    p["global_orient"][0, 1] = 0.35
    J_orient, uv_orient = make_case(p)

    rigid_distance_error = (
        torch.pdist(J_orient) - torch.pdist(J0)
    ).abs().max().item()

    assert rigid_distance_error < 1e-5
    assert (J_orient - J0).norm(dim=-1).max() > 1e-3

    # C. Elbow local rotation
    p = {k: v.clone() for k, v in base_params.items()}
    p["body_pose"][0, 53] = 0.75
    J_pose, uv_pose = make_case(p)

    wrist_change = (
        J_pose[20] - J0[20]
    ).norm().item()

    assert wrist_change > 1e-3

    cases = [
        ("transl", J_transl, uv_transl),
        ("global_orient", J_orient, uv_orient),
        ("body_pose_elbow", J_pose, uv_pose),
    ]

    rows = []

    for name, joints, uv in cases:
        change_3d = (
            (joints - J0).norm(dim=-1).mean().item()
        )
        change_2d = (
            (uv - uv0).norm(dim=-1).mean().item()
        )

        rows.append([name, change_3d, change_2d])

        print(
            f"{name}: "
            f"mean_3D={change_3d:.6f}, "
            f"mean_2D={change_2d:.4f}px"
        )

    print("Translation error:", transl_error)
    print("Rigid distance error:", rigid_distance_error)
    print("Left wrist displacement:", wrist_change)

    with (RESULTS / "d3_parameter_controls.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "control",
            "mean_3d_displacement",
            "mean_2d_displacement_px",
        ])
        writer.writerows(rows)

    plot_d3_parameter_controls(
        uv0.cpu().numpy(),
        {
            "Translation": uv_transl.cpu().numpy(),
            "Global Orientation": uv_orient.cpu().numpy(),
            "Elbow Pose": uv_pose.cpu().numpy(),
        },
        RESULTS / "d3_parameter_controls.png",
    )

    print("PASS")

def exp6_outlier_mask(data):
    print("\n===== D3 Exp 6: Outlier Mask =====")

    _, _, _, _, _, uv_gt, valid_gt = data

    uv_pred = (
        uv_gt + torch.tensor([5., -3.], dtype=DTYPE)
    ).detach().requires_grad_(True)

    uv_outlier = uv_gt.clone()
    uv_outlier[5] += torch.tensor(
        [300., -200.], dtype=DTYPE
    )

    mask = valid_gt.clone()
    mask[5] = False

    clean_loss = reprojection_loss(
        uv_pred, uv_gt, mask
    )
    masked_loss = reprojection_loss(
        uv_pred, uv_outlier, mask
    )
    unmasked_loss = reprojection_loss(
        uv_pred, uv_outlier, valid_gt
    )

    clean_grad = torch.autograd.grad(
        clean_loss, uv_pred, retain_graph=True
    )[0]

    masked_grad = torch.autograd.grad(
        masked_loss, uv_pred
    )[0]

    torch.testing.assert_close(
        masked_loss, clean_loss,
        atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        masked_grad, clean_grad,
        atol=1e-12, rtol=0
    )

    assert unmasked_loss > masked_loss * 10
    assert torch.isfinite(masked_grad).all()
    assert torch.all(masked_grad[5] == 0)

    print("Clean loss:", clean_loss.item())
    print("Masked loss:", masked_loss.item())
    print("Unmasked loss:", unmasked_loss.item())
    print("Mask gradient consistency: PASS")



def run_all():
    torch.manual_seed(42)
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = exp1_smpl_projection()
    exp2_loss_contract()
    exp3_camera_fitting(data)
    exp4_scale_depth_ambiguity(data)
    exp5_body_parameter_controls(data)
    exp6_outlier_mask(data)

    print("\n===== DAY 3 PHASE A: ALL CHECKS PASSED =====")


if __name__ == "__main__":
    run_all()
