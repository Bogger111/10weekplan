
import csv
import math
from pathlib import Path

import numpy as np
import torch

from week4.src.metrics import (
    mpjpe,
    root_aligned_mpjpe,
    pa_mpjpe,
)

DTYPE = torch.float64
ATOL = 1e-8
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def make_gt():
    """Two asymmetric, non-coplanar point sets."""
    return torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 1.2, 0.3],
            [0.2, 0.1, 1.5],
            [1.3, 0.5, 0.7],
        ],
        [
            [0.1, -0.2, 0.3],
            [1.4, 0.3, 0.0],
            [0.4, 1.1, 0.8],
            [0.7, -0.1, 1.7],
            [1.6, 1.2, 0.4],
        ],
    ], dtype=DTYPE)


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=DTYPE)


# ============================================================
# Independent reference: NumPy similarity Procrustes
# ============================================================

def numpy_pa_reference(pred, gt, mask=None):
    """
    Independent NumPy reference.
    Each batch sample is fitted independently.
    Returns global valid-joint-weighted PA-MPJPE.
    """
    P = pred.detach().cpu().numpy()
    G = gt.detach().cpu().numpy()

    B, J, _ = P.shape

    if mask is None:
        M = np.ones((B, J), dtype=bool)
    else:
        M = mask.detach().cpu().numpy()

    all_errors = []

    for b in range(B):
        p = P[b, M[b]]
        g = G[b, M[b]]

        if len(p) < 3:
            raise ValueError("Too few valid joints")

        mu_p = p.mean(axis=0, keepdims=True)
        mu_g = g.mean(axis=0, keepdims=True)

        pc = p - mu_p
        gc = g - mu_g

        H = pc.T @ gc
        U, singular, Vh = np.linalg.svd(H)

        D = np.eye(3)
        D[-1, -1] = -1.0 if np.linalg.det(U @ Vh) < 0 else 1.0

        R = U @ D @ Vh

        numerator = np.sum((pc @ R) * gc)
        denominator = np.sum(pc ** 2)

        if denominator <= 1e-12:
            raise ValueError("Degenerate point set")

        scale = numerator / denominator
        t = mu_g - scale * (mu_p @ R)

        aligned = scale * (p @ R) + t

        all_errors.extend(
            np.linalg.norm(aligned - g, axis=-1).tolist()
        )

    return float(np.mean(all_errors))


# ============================================================
# Assertions and reporting
# ============================================================

def close(actual, expected, label, atol=ATOL):
    assert math.isfinite(actual), (
        f"{label}: non-finite result {actual}"
    )
    assert abs(actual - expected) <= atol, (
        f"{label}: got {actual}, expected {expected}"
    )


def expect_value_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(
        f"{label}: expected ValueError, but no error was raised"
    )


def manual_root_masked(pred, gt, mask):
    """Direct formula independent of user's root function."""
    p = pred - pred[:, :1, :]
    g = gt - gt[:, :1, :]

    distances = torch.linalg.vector_norm(p - g, dim=-1)
    return distances[mask].mean().item()


def evaluate(name, pred, gt, mask=None):
    raw = mpjpe(pred, gt, mask).item()
    root = root_aligned_mpjpe(
        pred, gt, root_index=0, mask=mask
    ).item()
    pa = pa_mpjpe(pred, gt, mask).item()

    reference = numpy_pa_reference(pred, gt, mask)

    close(pa, reference, f"{name}: NumPy reference")

    print(
        f"{name:16s} | "
        f"Raw={raw:.8f} | "
        f"Root={root:.8f} | "
        f"PA={pa:.8f}"
    )

    return raw, root, pa, reference


def run_all():
    gt = make_gt()
    R = rotation_z(math.pi / 3)
    delta = torch.tensor([0.1, -0.2, 0.3], dtype=DTYPE)

    rows = []
    failures = []

    def run_case(name, pred, gt_case, check, mask=None):
        print(f"\n===== {name} =====")

        row = {
            "case": name,
            "status": "FAIL",
            "raw_mpjpe": "",
            "root_mpjpe": "",
            "pa_mpjpe": "",
            "numpy_pa_reference": "",
            "detail": "",
        }

        try:
            raw, root, pa, reference = evaluate(
                name, pred, gt_case, mask
            )

            row.update({
                "raw_mpjpe": raw,
                "root_mpjpe": root,
                "pa_mpjpe": pa,
                "numpy_pa_reference": reference,
            })

            check(raw, root, pa)
            row["status"] = "PASS"
            print("PASS")

        except Exception as exc:
            row["detail"] = f"{type(exc).__name__}: {exc}"
            failures.append(name)
            print("FAIL:", row["detail"])

        rows.append(row)

    def run_guard(name, fn):
        print(f"\n===== {name} =====")
        try:
            fn()
            print("PASS")
            rows.append({
                "case": name,
                "status": "PASS",
                "detail": "",
            })
        except Exception as exc:
            failures.append(name)
            print("FAIL:", exc)
            rows.append({
                "case": name,
                "status": "FAIL",
                "detail": f"{type(exc).__name__}: {exc}",
            })

    # --------------------------------------------------------
    # Phase A: Fundamental geometry
    # --------------------------------------------------------

    print("\n========== D4 PHASE A ==========")

    run_case(
        "Exp1 Identity",
        gt.clone(), gt,
        lambda raw, root, pa: (
            close(raw, 0, "Identity Raw"),
            close(root, 0, "Identity Root"),
            close(pa, 0, "Identity PA"),
        ),
    )

    pred_transl = gt + delta
    expected_translation = torch.linalg.vector_norm(delta).item()

    run_case(
        "Exp2 Translation",
        pred_transl, gt,
        lambda raw, root, pa: (
            close(raw, expected_translation, "Translation Raw"),
            close(root, 0, "Translation Root"),
            close(pa, 0, "Translation PA"),
        ),
    )

    pred_rotation = gt @ R

    def check_rotation(raw, root, pa):
        assert raw > 0.1
        assert root > 0.1
        close(pa, 0, "Rotation PA")

    run_case(
        "Exp3 Rotation",
        pred_rotation, gt, check_rotation,
    )

    pred_scale = gt * 2.0

    def check_scale(raw, root, pa):
        assert raw > 0.1
        assert root > 0.1
        close(pa, 0, "Scale PA")

    run_case(
        "Exp4 Scale",
        pred_scale, gt, check_scale,
    )

    # Reflection cannot be fitted by R with det(R) = +1.
    pred_reflection = gt * torch.tensor(
        [-1.0, 1.0, 1.0], dtype=DTYPE
    )

    def check_reflection(raw, root, pa):
        assert raw > 0
        assert root > 0
        assert pa > 1e-3, (
            "Reflection appears to have been removed!"
        )

    run_case(
        "Exp5 Reflection",
        pred_reflection, gt, check_reflection,
    )

    pred_local = gt.clone()
    pred_local[:, -1, 2] += 0.25

    def check_local(raw, root, pa):
        close(raw, 0.05, "Local Raw")
        close(root, 0.05, "Local Root")
        assert pa > 1e-3

    run_case(
        "Exp6 Local Error",
        pred_local, gt, check_local,
    )

    # Each sample requires a different similarity transform.
    pred_mixed = torch.stack([
        gt[0] + delta,
        (1.7 * (gt @ R) + delta)[1],
    ])

    run_case(
        "Exp7 Batch",
        pred_mixed, gt,
        lambda raw, root, pa: close(
            pa, 0, "Independent Batch PA"
        ),
    )

    # --------------------------------------------------------
    # Phase B: Mask / invalid inputs / degenerate inputs
    # --------------------------------------------------------

    print("\n========== D4 PHASE B ==========")

    mask = torch.tensor([
        [True, True, True, True, False],
        [True, True, False, True, True],
    ])

    # Corrupt excluded joints with NaNs on both sides.
    pred_masked = pred_mixed.clone()
    gt_masked = gt.clone()

    pred_masked[~mask] = float("nan")
    gt_masked[~mask] = float("nan")

    def check_masked(raw, root, pa):
        expected_raw = torch.linalg.vector_norm(
            pred_mixed - gt, dim=-1
        )[mask].mean().item()

        expected_root = manual_root_masked(
            pred_masked, gt_masked, mask
        )

        close(raw, expected_raw, "Masked Raw")
        close(root, expected_root, "Masked Root")
        close(pa, 0, "Masked PA")

    run_case(
        "Exp8 Mask + NaN",
        pred_masked, gt_masked,
        check_masked, mask,
    )

    # Root is excluded: root-based alignment has no valid anchor.
    bad_root_mask = torch.ones(
        gt.shape[:2], dtype=torch.bool
    )
    bad_root_mask[0, 0] = False

    run_guard(
        "Exp9 Invalid Root",
        lambda: expect_value_error(
            lambda: root_aligned_mpjpe(
                gt, gt, mask=bad_root_mask
            ),
            "Invalid Root",
        ),
    )

    # Completely collapsed prediction must not return a fake score.
    collapsed = torch.zeros_like(gt)

    run_guard(
        "Exp10 Collapsed",
        lambda: expect_value_error(
            lambda: pa_mpjpe(collapsed, gt),
            "Collapsed",
        ),
    )

    # Collinear points are insufficient for stable 3D rotation.
    line = torch.arange(
        gt.shape[1], dtype=DTYPE
    )[None, :, None]

    pred_line = torch.cat(
        [line, 2 * line, -line], dim=-1
    ).expand_as(gt).clone()

    run_guard(
        "Exp11 Collinear",
        lambda: expect_value_error(
            lambda: pa_mpjpe(pred_line, gt),
            "Collinear",
        ),
    )

    empty_mask = torch.zeros(
        gt.shape[:2], dtype=torch.bool
    )

    run_guard(
        "Exp12 Empty Mask",
        lambda: (
            expect_value_error(
                lambda: mpjpe(gt, gt, empty_mask),
                "Empty MPJPE mask",
            ),
            expect_value_error(
                lambda: pa_mpjpe(gt, gt, empty_mask),
                "Empty PA mask",
            ),
        ),
    )

    # Metric units must follow input units.
    def check_units():
        raw_m = mpjpe(pred_transl, gt).item()
        raw_mm = mpjpe(
            pred_transl * 1000,
            gt * 1000,
        ).item()
        close(raw_mm, raw_m * 1000, "Unit conversion", 1e-7)

    run_guard("Exp13 Units", check_units)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "d4_metrics.csv"

    fieldnames = [
        "case",
        "status",
        "raw_mpjpe",
        "root_mpjpe",
        "pa_mpjpe",
        "numpy_pa_reference",
        "detail",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = len(rows) - len(failures)

    print("\n========== D4 SUMMARY ==========")
    print(f"Passed: {passed}/{len(rows)}")
    print("Saved:", path)

    if failures:
        print("Failed cases:", ", ".join(failures))
        raise AssertionError(
            f"Day 4 has {len(failures)} failing checks"
        )

    print("DAY 4 METRICS: ALL CHECKS PASSED")


if __name__ == "__main__":
    run_all()
