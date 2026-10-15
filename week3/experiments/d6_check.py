import torch

from src.rotation import (
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_rotation_6D,
    rotation_6D_to_matrix,
)


DTYPE = torch.float64


def title(name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)


def rotation_distance(Ra, Rb):
    """
    Geodesic distance between two rotation matrices.
    return: radians
    """
    relative = Ra.transpose(-1, -2) @ Rb

    trace = (
        relative[..., 0, 0]
        + relative[..., 1, 1]
        + relative[..., 2, 2]
    )

    cos_delta = (trace - 1) / 2
    cos_delta = torch.clamp(cos_delta, -1.0, 1.0)

    return torch.acos(cos_delta)


# ============================================================
# Exp 1
# Same rotation represented by AA / Quaternion / 6D
# ============================================================

def exp1_multi_representation():
    title("EXP 1 | AA / Quaternion / 6D represent the same rotation")

    axis = torch.tensor([1., 2., 3.], dtype=DTYPE)
    axis = axis / torch.linalg.vector_norm(axis)

    theta = torch.tensor(1.1, dtype=DTYPE)

    # Axis-angle
    aa = theta * axis
    R_aa = axis_angle_to_matrix(aa)

    # Quaternion: (w, x, y, z)
    q = torch.cat([
        torch.cos(theta / 2).reshape(1),
        axis * torch.sin(theta / 2)
    ])

    R_q = quaternion_to_matrix(q)

    # 6D
    d6 = matrix_to_rotation_6D(R_aa)
    R_6d = rotation_6D_to_matrix(d6)

    print("AA:")
    print(aa)

    print("\nQuaternion:")
    print(q)

    print("\n6D:")
    print(d6)

    print("\nAA vs Quaternion max error:",
          torch.max(torch.abs(R_aa - R_q)))

    print("AA vs 6D max error:",
          torch.max(torch.abs(R_aa - R_6d)))

    print("Quaternion vs 6D max error:",
          torch.max(torch.abs(R_q - R_6d)))

    print("\nDtypes:")
    print("aa:", aa.dtype)
    print("q:", q.dtype)
    print("R_aa:", R_aa.dtype)
    print("R_q:", R_q.dtype)
    print("d6:", d6.dtype)
    print("R_6d:", R_6d.dtype)


# ============================================================
# Exp 2
# Quaternion double cover: q and -q
# ============================================================

def exp2_quaternion_sign():
    title("EXP 2 | Quaternion q and -q")

    theta = torch.tensor(torch.pi / 2, dtype=DTYPE)

    q = torch.stack([
        torch.cos(theta / 2),
        torch.tensor(0., dtype=DTYPE),
        torch.tensor(0., dtype=DTYPE),
        torch.sin(theta / 2),
    ])

    R1 = quaternion_to_matrix(q)
    R2 = quaternion_to_matrix(-q)

    parameter_distance = torch.linalg.vector_norm(q - (-q))
    matrix_error = torch.max(torch.abs(R1 - R2))

    print("q =", q)
    print("-q =", -q)

    print("\nparameter L2 =", parameter_distance)
    print("matrix max error =", matrix_error)


# ============================================================
# Exp 3
# Quaternion pi-boundary
# ============================================================

def exp3_quaternion_pi_boundary():
    title("EXP 3 | Canonical quaternion around pi boundary")

    angles_deg = [170, 175, 179, 181, 185, 190]

    quaternions = {}

    for deg in angles_deg:
        theta = torch.tensor(
            deg * torch.pi / 180,
            dtype=DTYPE
        )

        q = torch.stack([
            torch.cos(theta / 2),
            torch.tensor(0., dtype=DTYPE),
            torch.tensor(0., dtype=DTYPE),
            torch.sin(theta / 2),
        ])

        # canonicalization: force w >= 0
        if q[0] < 0:
            q = -q

        quaternions[deg] = q

        print(f"{deg:3d}° -> {q.tolist()}")

    q179 = quaternions[179]
    q181 = quaternions[181]

    parameter_distance = torch.linalg.vector_norm(q179 - q181)

    R179 = quaternion_to_matrix(q179)
    R181 = quaternion_to_matrix(q181)

    delta = rotation_distance(R179, R181)

    print("\n179° -> 181°")
    print("parameter distance:", parameter_distance)
    print("rotation distance rad:", delta)
    print("rotation distance deg:", torch.rad2deg(delta))


# ============================================================
# Exp 4
# 6D continuity across pi
# ============================================================

def exp4_6d_pi_boundary():
    title("EXP 4 | 6D representation around pi boundary")

    angles_deg = [170, 175, 179, 181, 185, 190]

    d6_list = {}

    for deg in angles_deg:
        theta = torch.tensor(
            deg * torch.pi / 180,
            dtype=DTYPE
        )

        q = torch.stack([
            torch.cos(theta / 2),
            torch.tensor(0., dtype=DTYPE),
            torch.tensor(0., dtype=DTYPE),
            torch.sin(theta / 2),
        ])

        R = quaternion_to_matrix(q)
        d6 = matrix_to_rotation_6D(R)

        d6_list[deg] = d6

        print(f"{deg:3d}° -> {d6.tolist()}")

    d179 = d6_list[179]
    d181 = d6_list[181]

    parameter_distance = torch.linalg.vector_norm(
        d179 - d181
    )

    print("\n179° -> 181°")
    print("6D parameter distance:", parameter_distance)


# ============================================================
# Exp 5
# Degenerate 6D inputs
# ============================================================

def exp5_degenerate_6d():
    title("EXP 5 | Degenerate 6D inputs")

    cases = {
        "zero_a1": torch.tensor(
            [0., 0., 0., 0., 1., 0.],
            dtype=DTYPE
        ),

        "collinear": torch.tensor(
            [1., 0., 0., 2., 0., 0.],
            dtype=DTYPE
        ),

        "near_collinear": torch.tensor(
            [1., 0., 0., 1., 1e-10, 0.],
            dtype=DTYPE
        ),

        "non_degenerate": torch.tensor(
            [1., 0., 0., 1., 1e-4, 0.],
            dtype=DTYPE
        ),
    }

    for name, d6 in cases.items():

        print(f"\n{name}")

        try:
            R = rotation_6D_to_matrix(d6)

            print("VALID")
            print(R)

            I = torch.eye(3, dtype=DTYPE)

            orth_error = torch.max(
                torch.abs(R @ R.T - I)
            )

            det = torch.det(R)

            print("orth error:", orth_error)
            print("det:", det)

        except ValueError as e:
            print("INVALID:", e)


# ============================================================
# Exp 6
# Row / column convention failure
# ============================================================

def exp6_row_column_convention():
    title("EXP 6 | Row / column convention failure")

    theta = torch.tensor(
        torch.pi / 2,
        dtype=DTYPE
    )

    q = torch.stack([
        torch.cos(theta / 2),
        torch.tensor(0., dtype=DTYPE),
        torch.tensor(0., dtype=DTYPE),
        torch.sin(theta / 2),
    ])

    R_ref = quaternion_to_matrix(q)

    d6 = matrix_to_rotation_6D(R_ref)

    R_row = rotation_6D_to_matrix(d6)

    # Deliberately wrong convention
    R_col = R_row.T

    I = torch.eye(3, dtype=DTYPE)

    print("R_ref:")
    print(R_ref)

    print("\nR_row:")
    print(R_row)

    print("\nR_col (wrong convention):")
    print(R_col)

    print("\nSO(3) checks")

    print(
        "row orth error:",
        torch.max(torch.abs(R_row @ R_row.T - I))
    )

    print(
        "col orth error:",
        torch.max(torch.abs(R_col @ R_col.T - I))
    )

    print("row det:", torch.det(R_row))
    print("col det:", torch.det(R_col))

    # Point rotation check
    p = torch.tensor(
        [1., 0., 0.],
        dtype=DTYPE
    )

    print("\nRotate p =", p)

    print("reference:", R_ref @ p)
    print("row:", R_row @ p)
    print("column mistake:", R_col @ p)


# ============================================================
# Exp 7
# Batch shape + numerical SO(3) check
# ============================================================

def exp7_batch_shape():
    title("EXP 7 | Batch shape")

    torch.manual_seed(42)

    d6 = torch.randn(
        2, 4, 5, 6,
        dtype=DTYPE
    )

    R = rotation_6D_to_matrix(d6)

    print("d6 shape:", d6.shape)
    print("R shape:", R.shape)

    I = torch.eye(
        3,
        dtype=DTYPE
    )

    orth_error = (
        R @ R.transpose(-1, -2) - I
    ).abs().amax()

    det_error = (
        torch.linalg.det(R) - 1
    ).abs().amax()

    print("max orth error:", orth_error)
    print("max det error:", det_error)
    print("all finite:", torch.isfinite(R).all())


# ============================================================
# Exp 8
# Backward through 6D decoder
# ============================================================

def exp8_backward():
    title("EXP 8 | Backward through 6D -> R")

    d6 = torch.tensor(
        [1.2, 0.3, -0.4,
         0.2, 1.1, 0.5],
        dtype=DTYPE,
        requires_grad=True
    )

    R = rotation_6D_to_matrix(d6)

    W = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]
    ], dtype=DTYPE)

    loss = (R * W).sum()

    loss.backward()

    print("R:")
    print(R)

    print("\nloss:", loss)
    print("grad:", d6.grad)

    print(
        "finite grad:",
        torch.isfinite(d6.grad).all()
    )


# ============================================================
# Exp 9
# R -> 6D -> R round-trip
# ============================================================

def exp9_round_trip():
    title("EXP 9 | R -> 6D -> R round-trip")

    axis = torch.tensor(
        [1., 2., 3.],
        dtype=DTYPE
    )

    axis = axis / torch.linalg.vector_norm(axis)

    theta = torch.tensor(
        0.7,
        dtype=DTYPE
    )

    q = torch.cat([
        torch.cos(theta / 2).reshape(1),
        axis * torch.sin(theta / 2)
    ])

    R = quaternion_to_matrix(q)

    d6 = matrix_to_rotation_6D(R)

    R_hat = rotation_6D_to_matrix(d6)

    I = torch.eye(
        3,
        dtype=DTYPE
    )

    print(
        "R round-trip error:",
        torch.max(torch.abs(R - R_hat))
    )

    print(
        "orth error:",
        torch.max(
            torch.abs(
                R_hat @ R_hat.T - I
            )
        )
    )

    print(
        "det:",
        torch.det(R_hat)
    )


# ============================================================
# Exp 10
# d6 -> R -> d6 is NOT parameter round-trip
# ============================================================

def exp10_d6_non_unique():
    title("EXP 10 | d6 -> R -> d6 does not preserve raw parameters")

    d6 = torch.tensor(
        [2., 0., 0.,
         1., 3., 0.],
        dtype=DTYPE
    )

    R = rotation_6D_to_matrix(d6)

    d6_hat = matrix_to_rotation_6D(R)

    R_hat = rotation_6D_to_matrix(d6_hat)

    print("original d6:")
    print(d6)

    print("\ncanonical d6 from R:")
    print(d6_hat)

    print(
        "\nparameter distance:",
        torch.linalg.vector_norm(
            d6 - d6_hat
        )
    )

    print(
        "rotation matrix error:",
        torch.max(
            torch.abs(R - R_hat)
        )
    )


# ============================================================
# Exp 11
# 64 random 6D samples
# ============================================================

def exp11_random_64():
    title("EXP 11 | 64 random 6D samples")

    torch.manual_seed(42)

    d6 = torch.randn(
        64, 6,
        dtype=DTYPE
    )

    R = rotation_6D_to_matrix(d6)

    I = torch.eye(
        3,
        dtype=DTYPE
    )

    orth_error = (
        R @ R.transpose(-1, -2) - I
    ).abs().amax(dim=(-2, -1))

    det_error = (
        torch.linalg.det(R) - 1
    ).abs()

    finite = torch.isfinite(
        R
    ).all(dim=(-2, -1))

    print(
        "max orth error:",
        orth_error.max()
    )

    print(
        "max det error:",
        det_error.max()
    )

    print(
        "all finite:",
        finite.all()
    )


# ============================================================
# Run all
# ============================================================

def run_all():

    experiments = [
        exp1_multi_representation,
        exp2_quaternion_sign,
        exp3_quaternion_pi_boundary,
        exp4_6d_pi_boundary,
        exp5_degenerate_6d,
        exp6_row_column_convention,
        exp7_batch_shape,
        exp8_backward,
        exp9_round_trip,
        exp10_d6_non_unique,
        exp11_random_64,
    ]

    for exp in experiments:
        exp()


if __name__ == "__main__":
    run_all()