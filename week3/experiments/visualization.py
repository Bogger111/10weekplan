import matplotlib.pyplot as plt
from pathlib import Path


JOINT_NAMES = [
    "root",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


def plot_skeleton(
    P,
    parents,
    save_path,
    title="Skeleton",
    joint_names=JOINT_NAMES,
    axis_lim=(-2, 2),
):
    P = P.detach().cpu()

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # =========================
    # joints
    # =========================

    ax.scatter(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        s=45
    )

    # joint labels
    for j, name in enumerate(joint_names):
        ax.text(
            P[j, 0].item(),
            P[j, 1].item(),
            P[j, 2].item(),
            f" {name}",
            fontsize=8
        )

    # =========================
    # bones
    # =========================

    for j in range(1, len(parents)):
        p = int(parents[j])

        xs = [
            P[p, 0].item(),
            P[j, 0].item()
        ]

        ys = [
            P[p, 1].item(),
            P[j, 1].item()
        ]

        zs = [
            P[p, 2].item(),
            P[j, 2].item()
        ]

        ax.plot(
            xs,
            ys,
            zs,
            linewidth=2
        )

    # =========================
    # axis
    # =========================

    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(*axis_lim)
    ax.set_ylim(*axis_lim)
    ax.set_zlim(*axis_lim)

    # 三轴同比例，非常重要
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved to: {save_path}")

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


JOINT_NAMES = [
    "root",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


def save_skeleton_animation(
    P_seq,
    parents,
    save_path,
    fps=30,
):
    P_seq = P_seq.detach().cpu()

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig = plt.figure(figsize=(7, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    def update(frame):

        ax.cla()

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(
            f"Forward Kinematics | Frame {frame:02d}"
        )

        P = P_seq[frame]

        # joints
        ax.scatter(
            P[:, 0],
            P[:, 1],
            P[:, 2],
            s=45
        )

        # bones
        for j in range(1, len(parents)):
            p = int(parents[j])

            ax.plot(
                [P[p, 0], P[j, 0]],
                [P[p, 1], P[j, 1]],
                [P[p, 2], P[j, 2]],
                linewidth=2
            )

        # names
        for j, name in enumerate(JOINT_NAMES):
            ax.text(
                P[j, 0].item(),
                P[j, 1].item(),
                P[j, 2].item(),
                f" {name}",
                fontsize=7
            )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(P_seq),
        interval=1000 / fps,
    )

    animation.save(
        save_path,
        writer=PillowWriter(fps=fps)
    )

    plt.close(fig)

    print("Saved:", save_path)

# ============================================================
# LBS Visualization
# ============================================================

def plot_lbs_comparison(
    vertices_rest,
    vertices_pose,
    save_path,
    title="LBS Comparison"
):

    from pathlib import Path
    import matplotlib.pyplot as plt

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rest = (
        vertices_rest
        .detach()
        .cpu()
        .numpy()
    )

    posed = (
        vertices_pose
        .detach()
        .cpu()
        .numpy()
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    # Rest vertices
    ax.scatter(
        rest[:, 0],
        rest[:, 1],
        s=12,
        alpha=0.4,
        label="Rest"
    )

    # Posed vertices
    ax.scatter(
        posed[:, 0],
        posed[:, 1],
        s=12,
        label="Posed"
    )

    # Rest strip edges
    ax.plot(
        rest[0::2, 0],
        rest[0::2, 1],
        "--",
        alpha=0.5
    )

    ax.plot(
        rest[1::2, 0],
        rest[1::2, 1],
        "--",
        alpha=0.5
    )

    # Posed strip edges
    ax.plot(
        posed[0::2, 0],
        posed[0::2, 1],
        linewidth=2
    )

    ax.plot(
        posed[1::2, 0],
        posed[1::2, 1],
        linewidth=2
    )

    # Elbow joint
    ax.scatter(
        [1.],
        [0.],
        marker="x",
        s=100,
        label="Elbow"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.8, 1.8)

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", save_path)

# ============================================================
# LBS Animation
# ============================================================

def save_lbs_animation(
    V_seq,
    save_path,
    fps=30
):

    from pathlib import Path

    import matplotlib.pyplot as plt

    from matplotlib.animation import (
        FuncAnimation,
        PillowWriter
    )

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    V_seq = (
        V_seq
        .detach()
        .cpu()
        .numpy()
    )

    num_frames = V_seq.shape[0]

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    def update(frame):

        ax.clear()

        V = V_seq[frame]

        upper = V[0::2]
        lower = V[1::2]

        ax.scatter(
            V[:, 0],
            V[:, 1],
            s=12
        )

        ax.plot(
            upper[:, 0],
            upper[:, 1],
            linewidth=2
        )

        ax.plot(
            lower[:, 0],
            lower[:, 1],
            linewidth=2
        )

        ax.scatter(
            [1.],
            [0.],
            marker="x",
            s=100
        )

        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-1.8, 1.8)

        ax.set_aspect("equal")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.set_title(
            f"LBS | Frame {frame:02d}"
        )

        ax.grid(True)

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps
    )

    animation.save(
        str(save_path),
        writer=PillowWriter(fps=fps)
    )

    plt.close(fig)

    print("Saved:", save_path)

# ============================================================
# Candy Wrapper Visualization
# ============================================================

def plot_lbs_candy_wrapper(
    V_blend,
    V_rigid,
    angles,
    save_path,
):

    from pathlib import Path
    import math
    import matplotlib.pyplot as plt

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    V_blend = V_blend.detach().cpu().numpy()
    V_rigid = V_rigid.detach().cpu().numpy()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    for i, theta in enumerate(angles):

        ax = axes[i]

        blend = V_blend[i]
        rigid = V_rigid[i]

        # Rigid transformation
        ax.plot(
            rigid[:, 1],
            rigid[:, 2],
            "o-",
            markersize=2,
            label="Rigid"
        )

        # Linear blend skinning
        ax.plot(
            blend[:, 1],
            blend[:, 2],
            "o-",
            markersize=2,
            label="LBS 50/50"
        )

        ax.set_title(
            f"Twist {math.degrees(theta):.0f}°"
        )

        ax.set_xlabel("Y")
        ax.set_ylabel("Z")

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)

        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()

    fig.suptitle(
        "Linear Blend Skinning: "
        "Candy-wrapper Artifact"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", save_path)