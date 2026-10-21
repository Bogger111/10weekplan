
import csv
from pathlib import Path

import numpy as np
import torch

from week3.src.smpl_runner import load_smpl_model, smpl_forward
from week4.src.camera import to_camera, project_perspective, project_weak, apply_crop_affine
from week4.experiments.visualization import plot_projection_grid


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "week4/results"
MODEL_PATH = ROOT / "week3/models/smpl_clean/SMPL_NEUTRAL.pkl"
DTYPE = torch.float64


def make_K(f=800.0):
    return torch.tensor([
        [f,   0.0, 320.0],
        [0.0, f,   240.0],
        [0.0, 0.0,   1.0],
    ], dtype=DTYPE)


# ============================================================
# Day 1 — Representation Bridge
# ============================================================

def exp1_tensor_shapes():
    print("\n===== Exp 1: Spatio-Temporal Shapes =====")

    B, T, J, C = 2, 9, 17, 32

    x = torch.arange(
        B * T * J * C, dtype=DTYPE
    ).reshape(B, T, J, C)

    spatial = x.reshape(B * T, J, C)
    spatial_back = spatial.reshape(B, T, J, C)

    temporal_joint = x.permute(0, 2, 1, 3).reshape(
        B * J, T, C
    )
    temporal_back = temporal_joint.reshape(
        B, J, T, C
    ).permute(0, 2, 1, 3)

    temporal_frame = x.reshape(B, T, J * C)
    frame_back = temporal_frame.reshape(B, T, J, C)

    assert torch.equal(x, spatial_back)
    assert torch.equal(x, temporal_back)
    assert torch.equal(x, frame_back)

    visible = torch.ones(B, T, J, dtype=torch.bool)
    visible[0, 3, 5] = False
    assert visible.sum().item() == B * T * J - 1

    print("Original:", x.shape)
    print("Spatial:", spatial.shape)
    print("Temporal per joint:", temporal_joint.shape)
    print("Temporal per frame:", temporal_frame.shape)
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
    assert J.shape[0] == 1 and J.shape[-1] == 3
    assert torch.isfinite(V).all()
    assert torch.isfinite(J).all()

    shifted = {k: v.clone() for k, v in params.items()}
    shifted["transl"][0, 0] = 0.1

    V_shift, J_shift = smpl_forward(model, **shifted)

    expected = torch.tensor([0.1, 0, 0], dtype=dtype)

    v_error = ((V_shift - V) - expected).abs().max().item()
    j_error = ((J_shift - J) - expected).abs().max().item()

    assert v_error < 1e-5
    assert j_error < 1e-5

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        RESULT_DIR / "body_reference.npz",
        vertices=V[0].cpu().numpy(),
        joints=J[0].cpu().numpy(),
        faces=np.asarray(model.faces),
        **{k: v.cpu().numpy() for k, v in params.items()},
    )

    print("Vertices:", V.shape)
    print("Joints:", J.shape)
    print("Translation vertex error:", v_error)
    print("Translation joint error:", j_error)
    print("PASS")


# ============================================================
# Day 2 — Camera Projection
# ============================================================

def exp3_perspective_sanity():
    print("\n===== Exp 3: Perspective Sanity =====")

    points = torch.tensor([
        [0.0, 0.0, 2.0],
        [0.1, 0.0, 2.0],
        [0.0, 0.1, 2.0],
        [0.1, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [float("nan"), 0.0, 2.0],
        [float("inf"), 0.0, 2.0],
        [0.0, 0.0, 1e-5],
    ], dtype=DTYPE)

    expected = torch.tensor([
        [320.0, 240.0],
        [360.0, 240.0],
        [320.0, 280.0],
        [340.0, 240.0],
    ], dtype=DTYPE)

    expected_valid = torch.tensor([
        True, True, True, True,
        False, False, False, False, False
    ])

    uv, valid = project_perspective(points, make_K())

    torch.testing.assert_close(
        uv[:4], expected, atol=1e-10, rtol=0
    )

    assert torch.equal(valid, expected_valid)
    assert torch.isnan(uv[~valid]).all()

    print("Valid pixels:\n", uv[valid])
    print("Valid count:", valid.sum().item())
    print("Max pixel error:", (uv[:4] - expected).abs().max().item())
    print("PASS")


def exp4_camera_extrinsics():
    print("\n===== Exp 4: Camera Extrinsics =====")

    R = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=DTYPE)

    t = torch.tensor([0.2, -0.1, 2.0], dtype=DTYPE)

    model_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.1, 0.5],
    ], dtype=DTYPE)

    expected_cam = torch.tensor([
        [0.2, -0.1, 2.0],
        [0.2,  0.9, 2.0],
        [-0.8, -0.1, 2.0],
        [0.1,  0.4, 2.5],
    ], dtype=DTYPE)

    expected_uv = torch.tensor([
        [400.0, 200.0],
        [400.0, 600.0],
        [0.0,   200.0],
        [352.0, 368.0],
    ], dtype=DTYPE)

    points_cam = to_camera(model_points, R, t)
    uv, valid = project_perspective(points_cam, make_K())

    cam_error = (points_cam - expected_cam).abs().max().item()
    pixel_error = (uv - expected_uv).abs().max().item()

    torch.testing.assert_close(
        points_cam, expected_cam, atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        uv, expected_uv, atol=1e-10, rtol=0
    )
    assert valid.all()

    print("Camera error:", cam_error)
    print("Pixel error:", pixel_error)
    print("PASS")


def exp5_focal_length():
    print("\n===== Exp 5: Focal Length =====")

    point = torch.tensor([[0.2, 0.1, 2.0]], dtype=DTYPE)

    rows = []
    plot_points = []

    for f in [400.0, 800.0, 1200.0]:
        uv, valid = project_perspective(point, make_K(f))

        expected = torch.tensor([
            [320.0 + 0.1 * f, 240.0 + 0.05 * f]
        ], dtype=DTYPE)

        torch.testing.assert_close(
            uv, expected, atol=1e-10, rtol=0
        )
        assert valid.all()

        u, v = uv[0].tolist()
        print(f"f={f:.0f}: ({u:.3f}, {v:.3f})")

        rows.append(["focal", f, 2.0, u, v])
        plot_points.append((f"f={int(f)}", (u, v)))

    print("PASS")
    return rows, plot_points


def exp6_depth():
    print("\n===== Exp 6: Depth =====")

    rows = []
    plot_points = []

    for z in [2.0, 4.0, 8.0]:
        point = torch.tensor([[0.2, 0.1, z]], dtype=DTYPE)
        uv, valid = project_perspective(point, make_K())

        expected = torch.tensor([
            [320.0 + 160.0 / z, 240.0 + 80.0 / z]
        ], dtype=DTYPE)

        torch.testing.assert_close(
            uv, expected, atol=1e-10, rtol=0
        )
        assert valid.all()

        u, v = uv[0].tolist()
        print(f"z={z:.0f}: ({u:.3f}, {v:.3f})")

        rows.append(["depth", 800.0, z, u, v])
        plot_points.append((f"z={int(z)}", (u, v)))

    print("PASS")
    return rows, plot_points


# ============================================================
# Day 2 Phase B — Batch / Weak / Crop
# ============================================================

def exp7_batch_camera():
    print("\n===== Exp 7: Batch Camera =====")

    points = torch.tensor([
        [[0., 0., 0.],
         [0.1, 0.2, 0.5],
         [0.2, 0., -3.]],

        [[0., 0., 0.],
         [1., 0., 0.],
         [0., 1., 0.5]],
    ], dtype=DTYPE)

    R = torch.stack([
        torch.eye(3, dtype=DTYPE),
        torch.tensor([
            [0., -1., 0.],
            [1.,  0., 0.],
            [0.,  0., 1.],
        ], dtype=DTYPE),
    ])

    t = torch.tensor([
        [0., 0., 2.],
        [0.2, -0.1, 3.],
    ], dtype=DTYPE)

    K = torch.stack([make_K(800.), make_K(600.)])

    expected_cam = torch.tensor([
        [[0., 0., 2.],
         [0.1, 0.2, 2.5],
         [0.2, 0., -1.]],

        [[0.2, -0.1, 3.],
         [0.2, 0.9, 3.],
         [-0.8, -0.1, 3.5]],
    ], dtype=DTYPE)

    expected_uv = torch.tensor([
        [[320., 240.],
         [352., 304.],
         [float("nan"), float("nan")]],

        [[360., 220.],
         [360., 420.],
         [320. - 480. / 3.5, 240. - 60. / 3.5]],
    ], dtype=DTYPE)

    expected_valid = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])

    points_cam = to_camera(points, R, t)
    uv, valid = project_perspective(points_cam, K)

    assert points_cam.shape == (2, 3, 3)
    assert uv.shape == (2, 3, 2)
    assert valid.shape == (2, 3)

    torch.testing.assert_close(
        points_cam, expected_cam, atol=1e-10, rtol=0
    )
    assert torch.equal(valid, expected_valid)

    torch.testing.assert_close(
        uv[valid], expected_uv[valid],
        atol=1e-10, rtol=0
    )
    assert torch.isnan(uv[~valid]).all()

    print("Batch camera coordinates: PASS")
    print("Batch projection: PASS")


def exp8_weak_perspective():
    print("\n===== Exp 8: Weak Perspective =====")

    points = torch.tensor([
        [0., 0., 2.],
        [0.2, 0.1, 2.],
        [0.2, 0.1, 4.],
        [0., 0., -1.],
        [float("inf"), 0., 2.],
    ], dtype=DTYPE)

    uv, valid = project_weak(
        points, make_K(), z_ref=4.0, z_min=1e-7
    )

    expected_uv = torch.tensor([
        [320., 240.],
        [360., 260.],
        [360., 260.],
    ], dtype=DTYPE)

    expected_valid = torch.tensor(
        [True, True, True, False, False]
    )

    assert torch.equal(valid, expected_valid)

    torch.testing.assert_close(
        uv[:3], expected_uv, atol=1e-10, rtol=0
    )

    assert torch.isnan(uv[~valid]).all()

    perspective_uv, _ = project_perspective(
        points, make_K()
    )

    # Equal reference depth gives equal projection.
    torch.testing.assert_close(
        uv[2], perspective_uv[2],
        atol=1e-10, rtol=0
    )

    # Different depth gives a different perspective result.
    assert not torch.allclose(
        uv[1], perspective_uv[1]
    )

    print("Weak UV:\n", uv[:3])
    print("Shared-depth behavior: PASS")
    print("Principal-point behavior: PASS")
    print("Invalid mask: PASS")


def exp9_crop_affine():
    print("\n===== Exp 9: Crop Affine =====")

    points_cam = torch.tensor([
        [0., 0., 2.],
        [0.2, 0.1, 2.],
    ], dtype=DTYPE)

    K = make_K()

    # Crop origin=(100,100), resize factors=(2,3).
    A = torch.tensor([
        [2., 0., -200.],
        [0., 3., -300.],
        [0., 0., 1.],
    ], dtype=DTYPE)

    uv, valid = project_perspective(points_cam, K)
    assert valid.all()

    uv_crop = apply_crop_affine(uv, A)

    expected = torch.tensor([
        [440., 420.],
        [600., 540.],
    ], dtype=DTYPE)

    assert uv_crop.shape == (2, 2)

    torch.testing.assert_close(
        uv_crop, expected, atol=1e-10, rtol=0
    )

    # Projection then crop == projection with updated K.
    K_crop = A @ K

    uv_direct, valid_direct = project_perspective(
        points_cam, K_crop
    )

    assert valid_direct.all()

    torch.testing.assert_close(
        uv_crop, uv_direct,
        atol=1e-10, rtol=0
    )

    # Independent affine transform for each batch item.
    uv_batch = torch.stack([uv, uv])
    A_batch = torch.stack([
        A,
        torch.eye(3, dtype=DTYPE),
    ])

    uv_batch_out = apply_crop_affine(
        uv_batch, A_batch
    )

    torch.testing.assert_close(
        uv_batch_out,
        torch.stack([expected, uv]),
        atol=1e-10,
        rtol=0,
    )

    print("Crop UV:\n", uv_crop)
    print("K' = A @ K: PASS")
    print("Batch crop: PASS")



# ============================================================
# Run All
# ============================================================

def run_all():
    torch.manual_seed(42)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Day 1
    exp1_tensor_shapes()
    exp2_smpl_reference()

    # Day 2
    exp3_perspective_sanity()
    exp4_camera_extrinsics()
    focal_rows, focal_points = exp5_focal_length()
    depth_rows, depth_points = exp6_depth()

    with (RESULT_DIR / "d2_projection.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "f", "z", "u", "v"])
        writer.writerows(focal_rows + depth_rows)

    plot_projection_grid(
        focal_points,
        depth_points,
        RESULT_DIR / "projection_grid.png",
    )

    exp7_batch_camera()
    exp8_weak_perspective()
    exp9_crop_affine()

    print("\n===== Week 04: CURRENT CHECKS PASSED =====")


if __name__ == "__main__":
    run_all()
