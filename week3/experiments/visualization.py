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