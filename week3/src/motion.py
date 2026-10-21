import torch
import math

def generate_elbow_motion(
        num_frames = 60,
        max_theta = math.pi / 2
):
    time_sequence = torch.linspace(0, 1, num_frames)

    rotation_sequence = max_theta / 2 * (
        1 - torch.cos(2 * math.pi * time_sequence)
    )

    body_pose = torch.zeros(num_frames, 69)
    body_pose[:, 53] = rotation_sequence

    return body_pose