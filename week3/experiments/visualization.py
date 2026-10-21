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


# ============================================================
# SMPL Mesh Visualization
# ============================================================

def plot_smpl_mesh_cases(
    vertices,
    faces,
    titles,
    save_path,
    face_stride=5,
):

    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()

    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()

    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)

    N = vertices.shape[0]
    assert len(titles) == N

    # SMPL commonly uses Y as the vertical direction.
    # Convert display axes to X, Z, Y for matplotlib.
    display_vertices = vertices[..., [0, 2, 1]]

    # Use the same sampled faces and camera for all cases.
    sampled_faces = faces[::face_stride]

    # Shared axis limits for meaningful comparisons.
    all_points = display_vertices.reshape(-1, 3)

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    center = (mins + maxs) / 2
    radius = max((maxs - mins).max() / 2 * 1.1, 0.1)

    fig = plt.figure(
        figsize=(5 * N, 6)
    )

    for i in range(N):

        ax = fig.add_subplot(
            1, N, i + 1,
            projection="3d",
        )

        V = display_vertices[i]

        triangles = V[sampled_faces]

        mesh = Poly3DCollection(
            triangles,
            facecolors="lightsteelblue",
            edgecolors="none",
            alpha=1.0,
        )

        ax.add_collection3d(mesh)

        ax.set_xlim(
            center[0] - radius,
            center[0] + radius,
        )

        ax.set_ylim(
            center[1] - radius,
            center[1] + radius,
        )

        ax.set_zlim(
            center[2] - radius,
            center[2] + radius,
        )

        ax.set_box_aspect((1, 1, 1))

        ax.view_init(
            elev=15,
            azim=-80,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

        ax.set_title(titles[i])

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", save_path)



def plot_smpl_pose_ablation(
    vertices_full,
    vertices_off,
    joints_full,
    faces,
    save_path,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from pathlib import Path

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    v_on = vertices_full[0].detach().cpu().numpy()
    v_off = vertices_off[0].detach().cpu().numpy()
    elbow = joints_full[0, 18].detach().cpu().numpy()

    faces = np.asarray(faces, dtype=np.int64)

    # Original SMPL Y-up -> Matplotlib Z-up.
    def display(v):
        return v[..., [0, 2, 1]]

    v_on = display(v_on)
    v_off = display(v_off)
    elbow = display(elbow)

    delta = np.linalg.norm(v_on - v_off, axis=-1)

    fig = plt.figure(figsize=(16, 6))
    axes = [
        fig.add_subplot(1, 3, i + 1, projection="3d")
        for i in range(3)
    ]

    all_points = np.concatenate([v_on, v_off], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(np.max(maxs - mins) * 0.55, 0.1)

    def set_axes(ax, c, r):
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-80)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y (up)")

    # Panel 1 and 2: complete human meshes
    for ax, vertices, title in zip(
        axes[:2],
        [v_off, v_on],
        ["Pose Correctives OFF", "Full SMPL"],
    ):
        triangles = vertices[faces]

        mesh = Poly3DCollection(
            triangles,
            facecolors="lightsteelblue",
            edgecolors="none",
            alpha=1.0,
        )

        ax.add_collection3d(mesh)
        set_axes(ax, center, radius)
        ax.set_title(title)

    # Panel 3: elbow displacement heatmap
    zoom_radius = 0.25
    triangles = v_on[faces]
    face_centers = triangles.mean(axis=1)

    near_faces = (
        np.linalg.norm(
            face_centers - elbow[None, :],
            axis=1,
        ) < zoom_radius
    )

    if not np.any(near_faces):
        raise ValueError("No mesh faces found near elbow")

    local_faces = faces[near_faces]
    local_triangles = v_on[local_faces]

    face_delta = delta[local_faces].mean(axis=1)

    norm = Normalize(
        vmin=0,
        vmax=max(float(delta.max()), 1e-8),
    )

    cmap = plt.get_cmap("viridis")

    heatmap_mesh = Poly3DCollection(
        local_triangles,
        facecolors=cmap(norm(face_delta)),
        edgecolors="none",
        alpha=1.0,
    )

    axes[2].add_collection3d(heatmap_mesh)
    set_axes(axes[2], elbow, zoom_radius)
    axes[2].set_title("Elbow: Vertex Displacement")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig.colorbar(
        sm,
        ax=axes[2],
        shrink=0.65,
        pad=0.10,
        label="Vertex displacement (SMPL units)",
    )

    fig.suptitle(
        "SMPL Pose Blend Shapes Ablation",
        fontsize=15,
    )

    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", save_path)



def plot_smpl_motion(
    vertices,
    joints,
    faces,
    save_dir,
    fps=12,
    face_stride=10,
):
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # SMPL Y-up -> Matplotlib Z-up
    V = vertices.detach().cpu().numpy()[..., [0, 2, 1]]
    J = joints.detach().cpu().numpy()[..., [0, 2, 1]]
    F = np.asarray(faces, dtype=np.int64)

    T = V.shape[0]
    sampled_faces = F[::face_stride]

    # Shared camera and bounds for all frames.
    all_points = V.reshape(-1, 3)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    center = (mins + maxs) / 2
    radius = max(
        float((maxs - mins).max()) * 0.55,
        0.1,
    )

    def setup_axes(ax):
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-80)

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y (up)")

    # --------------------------------------------------------
    # 1. SMPL Mesh GIF
    # --------------------------------------------------------
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        V[0][sampled_faces],
        facecolors="lightsteelblue",
        edgecolors="none",
        alpha=1.0,
    )

    ax.add_collection3d(mesh)
    setup_axes(ax)

    def update(frame):
        mesh.set_verts(V[frame][sampled_faces])
        ax.set_title(f"SMPL Motion | Frame {frame + 1}/{T}")
        return (mesh,)

    animation = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000 / fps,
        blit=False,
    )

    gif_path = save_dir / "elbow_motion.gif"

    animation.save(
        str(gif_path),
        writer=PillowWriter(fps=fps),
        dpi=100,
    )

    plt.close(fig)
    print("Saved:", gif_path)

    # --------------------------------------------------------
    # 2. Wrist trajectory (3D)
    # --------------------------------------------------------
    wrist = J[:, 20, :]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        wrist[:, 0],
        wrist[:, 1],
        wrist[:, 2],
        linewidth=2,
        label="Wrist trajectory",
    )

    ax.scatter(
        *wrist[0],
        s=60,
        label="Start / End",
    )

    ax.scatter(
        *wrist[T // 2],
        s=60,
        label="Maximum bend",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.set_title("SMPL Left Wrist Trajectory")
    ax.legend()

    trajectory_path = save_dir / "wrist_trajectory.png"

    fig.savefig(
        trajectory_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)
    print("Saved:", trajectory_path)
