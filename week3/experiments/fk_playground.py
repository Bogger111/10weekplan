import torch
import numpy as np

from experiments.data import(
    R_local,
    root_trans,
    parents,
    offsets
)

from src.rotation import axis_angle_to_matrix
from src.kinematics import forward_kinematics

def generate_fk_sequence(
        num_sequence = 60,
        fps = 30
):
    J = len(parents)

    R_seq = (
        torch.eye(3, dtype=torch.float64).repeat(num_sequence, J, 1, 1)
    )

    root_seq = torch.zeros(
        num_sequence,
        3,
        dtype=torch.float64
    )

    angles = []

    for k in range(num_sequence):
        s = k / (num_sequence - 1)

        theta = (
            torch.pi / 4
            * torch.sin(
                torch.tensor(
                    2 * torch.pi * s,
                    dtype=torch.float64
                )
            )
        )
        angles.append(theta)

        R_seq[k, 2] = axis_angle_to_matrix(
            torch.stack([
                torch.tensor(0., dtype=torch.float64),
                torch.tensor(0., dtype=torch.float64),
                theta                
            ])
        )

    angles = torch.stack(angles)

    R_global, P_global = forward_kinematics(
        R_seq,
        offsets,
        parents,
        root_seq
    )

    return {
        "angles": angles,
        "R_local": R_seq,
        "R_global": R_global,
        "P_global": P_global,
        "root_trans": root_seq,
        "fps": fps,
    }

from pathlib import Path


def save_sequence_npz(
    sequence,
    save_path="results/d3_fk_sequence.npz"
):
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        save_path,
        angles=sequence["angles"].detach().cpu().numpy(),
        R_local=sequence["R_local"].detach().cpu().numpy(),
        R_global=sequence["R_global"].detach().cpu().numpy(),
        P_global=sequence["P_global"].detach().cpu().numpy(),
        root_trans=sequence["root_trans"].detach().cpu().numpy(),
        offsets=offsets.detach().cpu().numpy(),
        parents=parents.detach().cpu().numpy(),
        fps=sequence["fps"],
    )

    print("Saved:", save_path)

from experiments.visualization import (
    save_skeleton_animation,
)


def run_fk_playground():

    sequence = generate_fk_sequence(
        num_sequence=60,
        fps=30
    )

    save_sequence_npz(
        sequence,
        "results/d3_fk_sequence.npz"
    )

    save_skeleton_animation(
        sequence["P_global"],
        parents,
        "results/skeleton_fk.gif",
        fps=sequence["fps"]
    )