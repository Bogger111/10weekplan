
from pathlib import Path

import matplotlib.pyplot as plt


def plot_projection_grid(focal_points, depth_points, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cases = [
        ("Changing Focal Length", focal_points),
        ("Changing Depth", depth_points),
    ]

    for ax, (title, points) in zip(axes, cases):
        ax.scatter([320], [240], marker="+", s=120,
                   label="Principal point")

        for label, (u, v) in points:
            ax.scatter(u, v, s=65)
            ax.annotate(
                label,
                (u, v),
                xytext=(7, -14),
                textcoords="offset points",
            )

        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # Image Y points downward.
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Pixel u")
        ax.set_ylabel("Pixel v")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", save_path)



def plot_d3_reprojection(
    uv_gt, uv_initial, uv_final, history, save_path
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]

    ax.scatter(
        uv_gt[:, 0], uv_gt[:, 1],
        s=45, label="Synthetic GT"
    )
    ax.scatter(
        uv_initial[:, 0], uv_initial[:, 1],
        s=45, marker="x", label="Initial"
    )
    ax.scatter(
        uv_final[:, 0], uv_final[:, 1],
        s=35, marker="+", label="Fitted"
    )

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("SMPL Joint Reprojection")
    ax.set_xlabel("u (pixel)")
    ax.set_ylabel("v (pixel)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history[:, 0], history[:, 2])
    ax.set_title("Camera Fitting")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Mean pixel error (px)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", save_path)


def plot_d3_parameter_controls(baseline, cases, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    all_points = np.concatenate(
        [baseline] + list(cases.values()), axis=0
    )
    xy_min = all_points.min(axis=0)
    xy_max = all_points.max(axis=0)
    padding = np.maximum((xy_max - xy_min) * 0.1, 10)

    for ax, (name, current) in zip(axes, cases.items()):
        ax.scatter(
            baseline[:, 0], baseline[:, 1],
            facecolors="none", edgecolors="C0",
            s=45, label="Baseline",
        )
        ax.scatter(
            current[:, 0], current[:, 1],
            marker="x", s=45, label="Perturbed",
        )

        for p0, p1 in zip(baseline, current):
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                alpha=0.25,
            )

        ax.set_xlim(
            xy_min[0] - padding[0],
            xy_max[0] + padding[0],
        )
        ax.set_ylim(
            xy_max[1] + padding[1],
            xy_min[1] - padding[1],
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name)
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)
