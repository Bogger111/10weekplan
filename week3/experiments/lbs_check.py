import math
import torch

from src.lbs import (
    my_linear_blend_skinning,
    linear_blend_skinning
)

from experiments.visualization import (
    plot_lbs_comparison,
    save_lbs_animation,
    plot_lbs_candy_wrapper
)


DTYPE = torch.float64


# ============================================================
# Helper
# Test data
# ============================================================

def rotation_z(theta):

    c = math.cos(theta)
    s = math.sin(theta)

    return torch.tensor([
        [c, -s, 0.],
        [s,  c, 0.],
        [0., 0., 1.],
    ], dtype=DTYPE)


def create_rest_transforms():
    """
    Two joints:

    joint 0: root  (0,0,0)
    joint 1: elbow (1,0,0)

    return:
        G_rest [2,4,4]
    """

    G_rest = torch.eye(
        4,
        dtype=DTYPE
    ).repeat(2, 1, 1)

    G_rest[1, :3, 3] = torch.tensor(
        [1., 0., 0.],
        dtype=DTYPE
    )

    return G_rest


def create_pose(
    angle=0.,
    translation=(0., 0., 0.)
):
    """
    Construct analytical global transforms.

    angle: elbow rotation around z-axis

    return:
        G_pose [1,1,2,4,4]
        G_rest [2,4,4]
    """

    G_rest = create_rest_transforms()

    G_pose = (
        G_rest[None, None]
        .clone()
    )

    translation = torch.tensor(
        translation,
        dtype=DTYPE
    )

    # Root translation
    G_pose[0, 0, 0, :3, 3] = translation

    # Elbow global rotation
    G_pose[0, 0, 1, :3, :3] = rotation_z(
        angle
    )

    # Elbow global position
    G_pose[0, 0, 1, :3, 3] = (
        G_rest[1, :3, 3] + translation
    )

    return G_pose, G_rest


def create_vertices():
    """
    V = 3

    vertex 0: root controlled
    vertex 1: elbow controlled
    vertex 2: mixed weights
    """

    vertices = torch.tensor([
        [0.5,  0.2, 0.],
        [1.5,  0.2, 0.],
        [1.2, -0.3, 0.],
    ], dtype=DTYPE)

    weights = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ], dtype=DTYPE)

    return vertices, weights


# ============================================================
# Exp 1
# Identity pose
# ============================================================

def exp1_identity():

    vertices, weights = create_vertices()

    G_pose, G_rest = create_pose(
        angle=0.
    )

    V = linear_blend_skinning(
        vertices,
        weights,
        G_pose,
        G_rest
    )

    expected = vertices[None, None]

    error = (
        V - expected
    ).abs().max()

    print("\n===== Exp 1: Identity =====")

    print("Output shape:", V.shape)
    print("Max error:", error)

    assert V.shape == (1, 1, 3, 3)

    torch.testing.assert_close(
        V,
        expected,
        atol=1e-10,
        rtol=0
    )

    print("PASS")


# ============================================================
# Exp 2
# Single joint rotation
# ============================================================

def exp2_single_joint():

    vertices, weights = create_vertices()

    G_pose, G_rest = create_pose(
        angle=math.pi / 2
    )

    V = linear_blend_skinning(
        vertices,
        weights,
        G_pose,
        G_rest
    )

    # Vertex 0: root controlled
    expected_root = vertices[0]

    # Vertex 1: elbow controlled
    #
    # c = (1,0,0)
    # v = (1.5,0.2,0)
    #
    # v' = c + R(v-c)
    #    = (0.8,0.5,0)

    expected_elbow = torch.tensor(
        [0.8, 0.5, 0.],
        dtype=DTYPE
    )

    print("\n===== Exp 2: Single Joint =====")

    print("Root vertex:", V[0, 0, 0])
    print("Elbow vertex:", V[0, 0, 1])

    torch.testing.assert_close(
        V[0, 0, 0],
        expected_root,
        atol=1e-10,
        rtol=0
    )

    torch.testing.assert_close(
        V[0, 0, 1],
        expected_elbow,
        atol=1e-10,
        rtol=0
    )

    print("PASS")


# ============================================================
# Exp 3
# Mixed skinning weights
# ============================================================

def exp3_mixed_weights():

    vertices, weights = create_vertices()

    G_pose, G_rest = create_pose(
        angle=math.pi / 2
    )

    V = linear_blend_skinning(
        vertices,
        weights,
        G_pose,
        G_rest
    )

    # Vertex 2:
    #
    # v = (1.2,-0.3,0)
    #
    # Root transform:
    # v_root = (1.2,-0.3,0)
    #
    # Elbow transform:
    # v_elbow = (1.3,0.2,0)
    #
    # Mixed:
    # v' = 0.5*v_root + 0.5*v_elbow

    expected = torch.tensor(
        [1.25, -0.05, 0.],
        dtype=DTYPE
    )

    print("\n===== Exp 3: Mixed Weights =====")

    print("Weights:", weights[2])

    print(
        "Blended vertex:",
        V[0, 0, 2]
    )

    print("Expected:", expected)

    torch.testing.assert_close(
        V[0, 0, 2],
        expected,
        atol=1e-10,
        rtol=0
    )

    print("PASS")


# ============================================================
# Exp 4
# Inverse bind correction
# ============================================================

def exp4_inverse_bind():

    vertices, weights = create_vertices()

    G_pose, G_rest = create_pose(
        angle=math.pi / 2
    )

    V_correct = linear_blend_skinning(
        vertices,
        weights,
        G_pose,
        G_rest
    )

    # Elbow-controlled vertex
    v = vertices[1]

    v_h = torch.cat([
        v,
        torch.ones(1, dtype=DTYPE)
    ])

    # Incorrect:
    # Directly apply G_pose without inverse bind
    V_wrong = (
        G_pose[0, 0, 1] @ v_h
    )[:3]

    V_expected = torch.tensor(
        [0.8, 0.5, 0.],
        dtype=DTYPE
    )

    print("\n===== Exp 4: Inverse Bind =====")

    print(
        "Correct:",
        V_correct[0, 0, 1]
    )

    print("Wrong:", V_wrong)

    print(
        "Difference:",
        torch.linalg.vector_norm(
            V_correct[0, 0, 1] - V_wrong
        )
    )

    torch.testing.assert_close(
        V_correct[0, 0, 1],
        V_expected,
        atol=1e-10,
        rtol=0
    )

    assert torch.linalg.vector_norm(
        V_correct[0, 0, 1] - V_wrong
    ) > 0.5

    print("PASS")


# ============================================================
# Exp 5
# Batch and root translation
# ============================================================

def exp5_batch():

    vertices, weights = create_vertices()

    G_rest = create_rest_transforms()

    B = 2
    T = 4
    J = 2

    G_pose = (
        G_rest[None, None]
        .expand(B, T, J, 4, 4)
        .clone()
    )

    angles = [
        0.,
        math.pi / 2,
        -math.pi / 4,
        math.pi,
    ]

    for b in range(B):

        for t in range(T):

            translation = torch.zeros(
                3,
                dtype=DTYPE
            )

            if b == 1:
                translation = torch.tensor(
                    [1., 2., 3.],
                    dtype=DTYPE
                )

            # Root
            G_pose[b, t, 0, :3, 3] = translation

            # Elbow rotation
            G_pose[b, t, 1, :3, :3] = rotation_z(
                angles[t]
            )

            # Elbow global position
            G_pose[b, t, 1, :3, 3] = (
                G_rest[1, :3, 3] + translation
            )

    V = linear_blend_skinning(
        vertices,
        weights,
        G_pose,
        G_rest
    )

    # Batch 1 is batch 0 shifted by (1,2,3)
    difference = V[1] - V[0]

    expected = torch.tensor(
        [1., 2., 3.],
        dtype=DTYPE
    ).expand_as(difference)

    print("\n===== Exp 5: Batch =====")

    print("Output shape:", V.shape)

    print(
        "Translation max error:",
        (difference - expected).abs().max()
    )

    assert V.shape == (2, 4, 3, 3)

    torch.testing.assert_close(
        difference,
        expected,
        atol=1e-10,
        rtol=0
    )

    print("PASS")


# ============================================================
# Exp 6
# Loop vs Einsum
# ============================================================

def exp6_reference():

    vertices, weights = create_vertices()

    G_pose, G_rest = create_pose(
        angle=math.pi / 3,
        translation=(1., 2., 3.)
    )

    V_loop = my_linear_blend_skinning(
        vertices,
        weights,
        G_pose,
        G_rest
    )

    V_einsum = linear_blend_skinning(
        vertices,
        weights,
        G_pose,
        G_rest
    )

    error = (
        V_loop - V_einsum
    ).abs().max()

    print("\n===== Exp 6: Reference =====")

    print("Loop shape:", V_loop.shape)
    print("Einsum shape:", V_einsum.shape)

    print("Max difference:", error)

    torch.testing.assert_close(
        V_loop,
        V_einsum,
        atol=1e-10,
        rtol=0
    )

    print("PASS")


# ============================================================
# Exp 7
# LBS visualization and animation
# ============================================================

def exp7_animation():

    # Construct a simple arm strip
    N = 40

    x = torch.linspace(
        0.2, 1.8, N,
        dtype=DTYPE
    )

    y_upper = torch.full_like(x, 0.12)
    y_lower = torch.full_like(x, -0.12)

    zeros = torch.zeros_like(x)

    upper = torch.stack([
        x, y_upper, zeros
    ], dim=-1)

    lower = torch.stack([
        x, y_lower, zeros
    ], dim=-1)

    # Interleaved vertices:
    # upper0, lower0, upper1, lower1, ...
    vertices = torch.stack(
        [upper, lower],
        dim=1
    ).reshape(-1, 3)

    # Smooth transition from root to elbow
    alpha = (
        (vertices[:, 0] - 0.6) / 0.8
    ).clamp(0., 1.)

    weights = torch.stack([
        1. - alpha,
        alpha
    ], dim=-1)

    G_rest = create_rest_transforms()

    num_frames = 60
    fps = 30

    G_pose = (
        G_rest[None, None]
        .expand(1, num_frames, 2, 4, 4)
        .clone()
    )

    for t in range(num_frames):

        s = t / (num_frames - 1)

        # 0 -> 90 -> 0 degrees
        theta = (
            math.pi / 4
            * (1. - math.cos(2 * math.pi * s))
        )

        G_pose[0, t, 1, :3, :3] = rotation_z(
            theta
        )

    V = linear_blend_skinning(
        vertices,
        weights,
        G_pose,
        G_rest
    )

    V_seq = V[0]

    print("\n===== Exp 7: Animation =====")

    print("Vertices shape:", vertices.shape)
    print("Weights shape:", weights.shape)
    print("Sequence shape:", V_seq.shape)

    assert V_seq.shape == (60, 80, 3)

    torch.testing.assert_close(
        V_seq[0],
        vertices,
        atol=1e-10,
        rtol=0
    )

    torch.testing.assert_close(
        V_seq[-1],
        vertices,
        atol=1e-10,
        rtol=0
    )

    # Static comparison
    plot_lbs_comparison(
        vertices,
        V_seq[num_frames // 2],
        "results/lbs/lbs_comparison.png",
        title="Linear Blend Skinning: Rest vs Bent"
    )

    # GIF
    save_lbs_animation(
        V_seq,
        "results/lbs/lbs_animation.gif",
        fps=fps
    )

    print("PASS")

# ============================================================
# Exp 8
# Candy-wrapper artifact
# ============================================================

def rotation_x(theta):

    c = math.cos(theta)
    s = math.sin(theta)

    return torch.tensor([
        [1., 0., 0.],
        [0., c, -s],
        [0., s,  c],
    ], dtype=DTYPE)


def exp8_candy_wrapper():

    print("\n===== Exp 8: Candy Wrapper =====")

    # --------------------------------------------------------
    # 1. Create circular cross-section
    # --------------------------------------------------------

    N = 64
    radius = 0.2

    phi = torch.linspace(
        0,
        2 * math.pi,
        N + 1,
        dtype=DTYPE
    )[:-1]

    # Ring center = elbow = (1, 0, 0)

    vertices = torch.stack([
        torch.ones_like(phi),
        radius * torch.cos(phi),
        radius * torch.sin(phi),
    ], dim=-1)

    # --------------------------------------------------------
    # 2. Skinning weights
    # --------------------------------------------------------

    # 50/50 blend
    weights_blend = torch.full(
        (N, 2),
        0.5,
        dtype=DTYPE
    )

    # Fully controlled by elbow
    weights_rigid = torch.zeros(
        N, 2,
        dtype=DTYPE
    )

    weights_rigid[:, 1] = 1.0

    # --------------------------------------------------------
    # 3. Rest / Pose transforms
    # --------------------------------------------------------

    G_rest = create_rest_transforms()

    angles = [
        0.,
        math.pi / 2,
        math.pi,
    ]

    T = len(angles)

    G_pose = (
        G_rest[None, None]
        .expand(1, T, 2, 4, 4)
        .clone()
    )

    for t, theta in enumerate(angles):

        G_pose[0, t, 1, :3, :3] = rotation_x(
            theta
        )

    # --------------------------------------------------------
    # 4. Run LBS
    # --------------------------------------------------------

    V_blend = linear_blend_skinning(
        vertices,
        weights_blend,
        G_pose,
        G_rest
    )

    V_rigid = linear_blend_skinning(
        vertices,
        weights_rigid,
        G_pose,
        G_rest
    )

    # [T, N, 3]
    V_blend = V_blend[0]
    V_rigid = V_rigid[0]

    # --------------------------------------------------------
    # 5. Measure radius
    # --------------------------------------------------------

    # Since the ring center is on the X axis,
    # radius = sqrt(y^2 + z^2)

    radius_blend = torch.linalg.vector_norm(
        V_blend[..., 1:],
        dim=-1
    ).mean(dim=-1)

    radius_rigid = torch.linalg.vector_norm(
        V_rigid[..., 1:],
        dim=-1
    ).mean(dim=-1)

    print("Original radius:", radius)

    for t, theta in enumerate(angles):

        print(
            f"Angle {math.degrees(theta):.0f}°:"
        )

        print(
            "  LBS radius:",
            radius_blend[t].item()
        )

        print(
            "  Rigid radius:",
            radius_rigid[t].item()
        )

    # --------------------------------------------------------
    # 6. Analytical checks
    # --------------------------------------------------------

    expected_blend = torch.tensor([
        radius,
        radius / math.sqrt(2),
        0.,
    ], dtype=DTYPE)

    expected_rigid = torch.full(
        (T,),
        radius,
        dtype=DTYPE
    )

    torch.testing.assert_close(
        radius_blend,
        expected_blend,
        atol=1e-10,
        rtol=0
    )

    torch.testing.assert_close(
        radius_rigid,
        expected_rigid,
        atol=1e-10,
        rtol=0
    )

    # --------------------------------------------------------
    # 7. Visualization
    # --------------------------------------------------------

    plot_lbs_candy_wrapper(
        V_blend,
        V_rigid,
        angles,
        "results/lbs/candy_wrapper.png",
    )

    print("PASS")


# ============================================================
# Run all
# ============================================================

def run_all():

    exp1_identity()

    exp2_single_joint()

    exp3_mixed_weights()

    exp4_inverse_bind()

    exp5_batch()

    exp6_reference()

    exp7_animation()


if __name__ == "__main__":
    run_all()