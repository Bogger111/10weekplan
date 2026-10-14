import math

import torch

from src.rotation import (
    skew,
    axis_angle_to_matrix,
    apply_rotation,
    check_so3,
)


# ============================================================
# 1. Direction + inverse
# ============================================================

def exp_direction_inverse():
    print("\n=== Experiment 1: Direction & Inverse ===")

    omega = torch.tensor([
        0.,
        0.,
        math.pi / 2
    ])

    R = axis_angle_to_matrix(omega)

    p = torch.tensor([
        [1., 0., 0.]
    ])

    expected = torch.tensor([
        [0., 1., 0.]
    ])

    # Active rotation:
    #
    # e_x --Rz(pi/2)--> e_y
    #
    # Coordinate frame is fixed.
    # The vector itself is being rotated.
    p_rot = apply_rotation(R, p)

    # Inverse rotation:
    #
    # R^-1 = R^T
    #
    # e_y --R^T--> e_x
    p_rec = apply_rotation(
        R.transpose(-1, -2),
        p_rot
    )

    err_forward = (
        p_rot - expected
    ).abs().max()

    err_roundtrip = (
        p_rec - p
    ).abs().max()

    print("original:", p)
    print("rotated :", p_rot)
    print("expected:", expected)
    print("recovered:", p_rec)

    print(
        "forward error:",
        err_forward.item()
    )

    print(
        "round-trip error:",
        err_roundtrip.item()
    )


# ============================================================
# 2. Rotation composition / non-commutativity
# ============================================================

def exp_noncommutativity():
    print("\n=== Experiment 2: Non-commutativity ===")

    Rx = axis_angle_to_matrix(
        torch.tensor([
            math.pi / 2,
            0.,
            0.
        ])
    )

    Rz = axis_angle_to_matrix(
        torch.tensor([
            0.,
            0.,
            math.pi / 2
        ])
    )

    p = torch.tensor([
        [1., 0., 0.]
    ])

    # Matrix on the right acts first:
    #
    # R1 p = Rx (Rz p)
    R1 = Rx @ Rz

    # R2 p = Rz (Rx p)
    R2 = Rz @ Rx

    p1 = apply_rotation(R1, p)
    p2 = apply_rotation(R2, p)

    matrix_diff = (
        R1 - R2
    ).abs().max()

    point_diff = torch.linalg.norm(
        p1 - p2
    )

    print("Rx @ Rz result:", p1)
    print("Rz @ Rx result:", p2)

    print(
        "matrix max difference:",
        matrix_diff.item()
    )

    print(
        "point difference:",
        point_diff.item()
    )

    print(
        "commutative:",
        torch.allclose(R1, R2)
    )


# ============================================================
# 3. Length / distance preservation + reflection
# ============================================================

def exp_length_reflection():
    print("\n=== Experiment 3: Rotation vs Reflection ===")

    torch.manual_seed(42)

    points = torch.randn(
        8,
        3
    )

    R = axis_angle_to_matrix(
        torch.tensor([
            0.3,
            -0.2,
            0.4
        ])
    )

    # Reflection about yz-plane
    M = torch.diag(
        torch.tensor([
            -1.,
             1.,
             1.
        ])
    )

    # --------------------------------------------------------
    # Point norms
    # --------------------------------------------------------

    length_before = torch.linalg.norm(
        points,
        dim=-1
    )

    points_R = apply_rotation(
        R,
        points
    )

    points_M = apply_rotation(
        M,
        points
    )

    length_R = torch.linalg.norm(
        points_R,
        dim=-1
    )

    length_M = torch.linalg.norm(
        points_M,
        dim=-1
    )

    length_error_R = (
        length_R - length_before
    ).abs().max()

    length_error_M = (
        length_M - length_before
    ).abs().max()

    # --------------------------------------------------------
    # Pairwise distances
    # --------------------------------------------------------

    D_before = torch.cdist(
        points,
        points
    )

    D_R = torch.cdist(
        points_R,
        points_R
    )

    D_M = torch.cdist(
        points_M,
        points_M
    )

    distance_error_R = (
        D_R - D_before
    ).abs().max()

    distance_error_M = (
        D_M - D_before
    ).abs().max()

    print(
        "rotation length error:",
        length_error_R.item()
    )

    print(
        "reflection length error:",
        length_error_M.item()
    )

    print(
        "rotation distance error:",
        distance_error_R.item()
    )

    print(
        "reflection distance error:",
        distance_error_M.item()
    )

    print(
        "det(R):",
        torch.linalg.det(R).item()
    )

    print(
        "det(M):",
        torch.linalg.det(M).item()
    )

    print(
        "R in SO(3):",
        check_so3(R)
    )

    print(
        "M in SO(3):",
        check_so3(M)
    )


# ============================================================
# 4. Zero / small angle
# ============================================================

def exp_small_angle():
    print("\n=== Experiment 4: Zero & Small Angle ===")

    omegas = [
        torch.tensor([
            0.,
            0.,
            0.
        ]),

        torch.tensor([
            1e-8,
            0.,
            0.
        ]),

        torch.tensor([
            0.3,
            -0.2,
            0.4
        ]),
    ]

    I = torch.eye(3)

    for omega in omegas:

        R = axis_angle_to_matrix(
            omega
        )

        so3 = check_so3(R)

        identity_error = (
            R - I
        ).abs().max()

        print("\nomega:", omega)

        print(
            "R:\n",
            R
        )

        print(
            "finite:",
            torch.isfinite(R).all().item()
        )

        print(
            "identity error:",
            identity_error.item()
        )

        print(
            "orth error:",
            so3["orth_error"]
        )

        print(
            "det error:",
            so3["det_error"]
        )


# ============================================================
# 5. Gradient check
# ============================================================

def exp_gradient():
    print("\n=== Experiment 5: Gradient ===")

    weights = torch.tensor([
        [ 0.2, -0.3,  0.5],
        [ 0.7,  0.1, -0.4],
        [ 0.6, -0.2,  0.9]
    ])

    test_omegas = [
        [0.3, -0.2, 0.4],
        [1e-8, 0., 0.],
        [0., 0., 0.],
    ]

    for values in test_omegas:

        omega = torch.tensor(
            values,
            dtype=torch.float32,
            requires_grad=True
        )

        R = axis_angle_to_matrix(
            omega
        )

        # Do NOT use ||R||^2.
        #
        # For R in SO(3):
        #
        # ||R||_F^2 = 3
        #
        # which is constant.
        scalar = (
            R * weights
        ).sum()

        scalar.backward()

        print("\nomega:", omega.detach())

        print(
            "scalar:",
            scalar.item()
        )

        print(
            "grad:",
            omega.grad
        )

        print(
            "forward finite:",
            torch.isfinite(R).all().item()
        )

        print(
            "gradient finite:",
            torch.isfinite(
                omega.grad
            ).all().item()
        )


# ============================================================
# 6. Deliberately unsafe zero-angle implementation
# ============================================================

def axis_angle_to_matrix_unsafe(omega):
    """
    Deliberately unsafe implementation
    for the failure experiment.
    """

    theta = torch.linalg.norm(
        omega
    )

    K = skew(omega)

    I = torch.eye(
        3,
        dtype=omega.dtype,
        device=omega.device
    )

    A = (
        torch.sin(theta)
        / theta
    )

    B = (
        1 - torch.cos(theta)
    ) / theta**2

    return (
        I
        + A * K
        + B * (K @ K)
    )


def exp_unsafe_zero():
    print("\n=== Experiment 6: Unsafe Zero Angle ===")

    omega = torch.zeros(
        3,
        requires_grad=True
    )

    R = axis_angle_to_matrix_unsafe(
        omega
    )

    print("unsafe R:\n", R)

    print(
        "finite:",
        torch.isfinite(R).all().item()
    )


# ============================================================
# Run
# ============================================================

def run():

    exp_direction_inverse()

    exp_noncommutativity()

    exp_length_reflection()

    exp_small_angle()

    exp_gradient()

    exp_unsafe_zero()


if __name__ == "__main__":
    run()