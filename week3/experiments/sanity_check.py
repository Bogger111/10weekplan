import torch

from experiments.data import (
    R_local,
    offsets,
    parents,
    root_trans,
)

from experiments.visualization import (
    plot_skeleton,
)

from src.rotation import (
    axis_angle_to_matrix,
)

from src.kinematics import (
    forward_kinematics,
    forward_kinematics_reference
)


DTYPE = torch.float64


# ============================================================
# Exp 1
# Rest pose
# ============================================================

def exp1_rest_pose():

    _, P = forward_kinematics(
        R_local,
        offsets,
        parents,
        root_trans
    )

    print("\n===== Rest Pose =====")
    print(P)

    plot_skeleton(
        P,
        parents,
        "results/rest_pose.png",
        title="Rest Pose"
    )


# ============================================================
# Exp 2
# Left elbow Rz(90°)
# ============================================================

def exp2_elbow_rotation():

    R_test = R_local.clone()

    R_test[2] = axis_angle_to_matrix(
        torch.tensor(
            [0., 0., torch.pi / 2],
            dtype=DTYPE
        )
    )

    _, P = forward_kinematics(
        R_test,
        offsets,
        parents,
        root_trans
    )

    print("\n===== Left Elbow Rz(90°) =====")
    print(P)

    plot_skeleton(
        P,
        parents,
        "results/elbow_rz90.png",
        title="Left Elbow Rz(90°)"
    )


# ============================================================
# Exp 3
# Root translation
# ============================================================

def exp3_root_translation():

    root_trans_test = torch.tensor(
        [1., 2., 3.],
        dtype=DTYPE
    )

    _, P = forward_kinematics(
        R_local,
        offsets,
        parents,
        root_trans_test
    )

    print("\n===== Root Translation =====")
    print(P)

    plot_skeleton(
        P,
        parents,
        "results/root_translation.png",
        title="Root Translation (1,2,3)",
        axis_lim=(-3, 5)
    )


# ============================================================
# Exp 4
# Root Rz(90°)
# ============================================================

def exp4_root_rotation():

    R_test = R_local.clone()

    R_test[0] = axis_angle_to_matrix(
        torch.tensor(
            [0., 0., torch.pi / 2],
            dtype=DTYPE
        )
    )

    _, P = forward_kinematics(
        R_test,
        offsets,
        parents,
        root_trans
    )

    print("\n===== Root Rz(90°) =====")
    print(P)

    plot_skeleton(
        P,
        parents,
        "results/root_rz90.png",
        title="Root Rz(90°)"
    )


# ============================================================
# Exp 5
# Shoulder Rx(90°) + Elbow Rz(90°)
# ============================================================

def exp5_local_global():

    R_test = R_local.clone()

    # left shoulder
    R_test[1] = axis_angle_to_matrix(
        torch.tensor(
            [torch.pi / 2, 0., 0.],
            dtype=DTYPE
        )
    )

    # left elbow
    R_test[2] = axis_angle_to_matrix(
        torch.tensor(
            [0., 0., torch.pi / 2],
            dtype=DTYPE
        )
    )

    _, P = forward_kinematics(
        R_test,
        offsets,
        parents,
        root_trans
    )

    print(
        "\n===== Shoulder Rx(90°) + "
        "Elbow Rz(90°) ====="
    )

    print(P)

    plot_skeleton(
        P,
        parents,
        "results/local_global_test.png",
        title=(
            "Shoulder Rx(90°) + "
            "Elbow Rz(90°)"
        )
    )


def exp6_batch_reference():

    B = 2
    T = 4
    J = 7

    R_batch = torch.eye(
        3,
        dtype=torch.float64
    ).repeat(B, T, J, 1, 1)

    root_batch = torch.zeros(
        B, T, 3,
        dtype=torch.float64
    )

    # 第二个 batch 整体平移
    root_batch[1] = torch.tensor(
        [1., 2., 3.],
        dtype=torch.float64
    )

    R1, P1 = forward_kinematics(
        R_batch,
        offsets,
        parents,
        root_batch
    )

    R2, P2 = forward_kinematics_reference(
        R_batch,
        offsets,
        parents,
        root_batch
    )

    print("\n===== Batch + Reference =====")

    print("R shape:", R1.shape)
    print("P shape:", P1.shape)

    print(
        "R reference error:",
        (R1 - R2).abs().max()
    )

    print(
        "P reference error:",
        (P1 - P2).abs().max()
    )

    # 第二个 batch 和第一个 batch 的位置差
    print(
        "batch translation difference:"
    )

    print(
        P1[1] - P1[0]
    )

def exp7_bone_length():

    def check_bone_lengths(P, offsets, parents):
        """
        P: [..., J, 3]
        offsets: [J, 3]
        parents: [J]

        return:
            max bone length error
        """

        errors = []

        for j in range(1, len(parents)):
            p = int(parents[j])

            # 实际骨长
            actual_length = torch.linalg.vector_norm(
                P[..., j, :] - P[..., p, :],
                dim=-1
            )

            # rest pose 中定义的骨长
            expected_length = torch.linalg.vector_norm(
                offsets[j]
            )

            error = torch.abs(
                actual_length - expected_length
            )

            errors.append(error)

        errors = torch.stack(errors, dim=-1)

        return errors.max()

    R_test = R_local.clone()

    # shoulder Rx(90°)
    R_test[1] = axis_angle_to_matrix(
        torch.tensor(
            [torch.pi / 2, 0., 0.],
            dtype=torch.float64
        )
    )

    # elbow Rz(90°)
    R_test[2] = axis_angle_to_matrix(
        torch.tensor(
            [0., 0., torch.pi / 2],
            dtype=torch.float64
        )
    )

    _, P = forward_kinematics(
        R_test,
        offsets,
        parents,
        root_trans
    )

    error = check_bone_lengths(
        P,
        offsets,
        parents
    )

    print("\n===== Bone Length =====")
    print("max bone length error:", error)

# ============================================================
# Run all
# ============================================================

def run_all():

    exp1_rest_pose()

    exp2_elbow_rotation()

    exp3_root_translation()

    exp4_root_rotation()

    exp5_local_global()

    exp6_batch_reference()

    exp7_bone_length()
