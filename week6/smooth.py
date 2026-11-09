from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
import pickle


original_dir = Path(
    "/home/bogger/10weekplan/week6/results/"
    "blender_baseline_20261029_203921/video/anny_params"
)

output_dir = Path(
    "/home/bogger/10weekplan/week6/results/anny_smoothed_full"
)

files = sorted(original_dir.glob('*.pkl'))

all_positions = []
all_rotations = []
all_shapes = []
all_indices = []

for t, file in enumerate(files):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    pose = data['pose_parameters']
    shape = data['shape']

    root = pose[:, 0, :3, 3]
    x = root[:, 0]

    indices = torch.argsort(x, descending=True)

    ordered_root = root[indices]
    ordered_rotation = pose[indices, :, :3, :3]
    ordered_shape = shape[indices]


    all_positions.append(ordered_root.cpu().numpy())
    all_rotations.append(ordered_rotation.cpu().numpy())
    all_shapes.append(ordered_shape.cpu().numpy())
    all_indices.append(indices.cpu().tolist())

all_positions = np.stack(all_positions, axis=0)
all_rotations = np.stack(all_rotations, axis=0)
all_shapes = np.stack(all_shapes, axis=0)


def root_smooth(sequence, fps, cutoff=None):
    if cutoff is None:
        cutoff = fps * 0.2

    sos = butter(
        N=2,
        Wn=cutoff,
        btype='lowpass',
        output='sos',
        fs=fps
    )

    return sosfiltfilt(sos, sequence, axis=0)


def interpolate_rotation(r1, r2, alpha):
    relative = r1.inv() * r2
    rotvec = relative.as_rotvec()

    return r1 * R.from_rotvec(alpha * rotvec)


def so3_smooth(sequence, alpha=0.3):
    rotations = R.from_matrix(sequence)
    T = len(sequence)

    forward = [rotations[0]]

    for t in range(1, T):
        forward.append(
            interpolate_rotation(
                forward[-1],
                rotations[t],
                alpha
            )
        )

    backward = [None] * T
    backward[-1] = forward[-1]

    for t in range(T-2, -1, -1):
        backward[t] = interpolate_rotation(
            backward[t + 1],
            forward[t],
            alpha
        )

    return np.stack([r.as_matrix() for r in backward])




smoothed_positions = root_smooth(all_positions, 10, 2.0)
smoothed_shapes = root_smooth(all_shapes, 10, 1.0)

smoothed_rotations = np.empty_like(all_rotations)

for person_id in range(2):
    for joint_id in range(163):
        sequence = all_rotations[:, person_id, joint_id]

        smoothed_rotations[:, person_id, joint_id] = (
            so3_smooth(sequence, alpha=0.3)
        )




output_dir.mkdir(parents=True, exist_ok=True)

for t, file in enumerate(files):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    new_pose = data['pose_parameters'].clone()
    new_shape = data['shape'].clone()

    for person_id in range(2):
        original_index = all_indices[t][person_id]

        new_pose[original_index, 0, :3, 3] = torch.as_tensor(
            smoothed_positions[t, person_id].copy(),
            dtype=new_pose.dtype,
            device=new_pose.device
        )

        new_pose[original_index, :, :3, :3] = torch.as_tensor(
            smoothed_rotations[t, person_id].copy(),
            dtype=new_pose.dtype,
            device=new_pose.device
        )

        new_shape[original_index] = torch.as_tensor(
            smoothed_shapes[t, person_id].copy(),
            dtype=new_shape.dtype,
            device=new_shape.device
        )

    data['pose_parameters'] = new_pose
    data['shape'] = new_shape

    with open(output_dir / file.name, 'wb') as f:
        pickle.dump(data, f)

print(f"Finished: {len(files)} frames")
print(f"Saved to: {output_dir}")
