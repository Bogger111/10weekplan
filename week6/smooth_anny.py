from pathlib import Path
import pickle
import torch
import numpy as np

import matplotlib.pyplot as plt

para_dir = Path('/home/bogger/10weekplan/week6/results/blender_baseline_20261029_203921/video/anny_params')

files = sorted(para_dir.glob('*.pkl'))

all_positions = []

all_indices = []

for file in files:
    with open(file, 'rb') as f:
        data = pickle.load(f)

    pose = data['pose_parameters']


    root = pose[:, 0, :3, 3]

    x = root[:, 0]

    indices = torch.argsort(x, descending=True)

    ordered_root = root[indices]

    all_positions.append(ordered_root)
    all_indices.append(indices)
positions = torch.stack(all_positions, dim=0)

print(positions.shape)

from scipy.signal import butter, sosfiltfilt

positions_np = positions.detach().cpu().numpy()

fps = 10

def measure_displacement(original, filtered):
    difference = filtered - original
    squared_distance = np.sum(difference ** 2, axis=-1)
    return np.sqrt(np.mean(squared_distance))

def measure_jitter(x):
    acceleration = np.diff(
        x,
        n=2,
        axis=0
    )

    return np.linalg.norm(
        acceleration,
        axis=-1
    ).mean()

def check_fliter(position_np, cutoff):

    print('============================')
    print('cutoff',cutoff)

    sos = butter(
        N=2,
        Wn=cutoff,
        btype='lowpass',
        fs=fps,
        output='sos'
    )

    smoothed = sosfiltfilt(
        sos,
        positions_np,
        axis=0
    )

    print("Before:", measure_jitter(positions_np))
    print("After:", measure_jitter(smoothed))

    print('RMSE:',measure_displacement(positions_np, smoothed))



def check_person(positions_np, cutoff):
    sos = butter(
        N=2,
        Wn=cutoff,
        btype='lowpass',
        output='sos',
        fs=fps
    )

    smoothed = sosfiltfilt(sos, positions_np, axis=0)

    for person_id in range(2):
        original = positions_np[:, person_id, :]
        flitered = smoothed[:, person_id, :]

        print("=================")
        print('Person:',person_id,'cutoff:',cutoff)

        print('Before:',measure_jitter(original))
        print('After:',measure_jitter(flitered))

        print('RMSE:',measure_displacement(original, flitered))

    return smoothed

def do_filiter(original, cutoff):
    sos = butter(
        N=2,
        Wn=cutoff,
        btype='lowpass',
        output='sos',
        fs=fps
    )


    smoothed = sosfiltfilt(sos, original, axis=0)

    return smoothed

def main():
    smoothed = do_filiter(positions_np, 2.0)

    smoothed = torch.from_numpy(smoothed.copy())

    output_dir = Path(
    '/home/bogger/10weekplan/week6/results/anny_smoothed'
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for t, file in enumerate(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        new_pose = data['pose_parameters'].clone()

        for person_id in range(2):
            original_index = all_indices[t][person_id]


            new_pose[original_index, 0, :3, 3] = smoothed[t, person_id, :].to(dtype=new_pose.dtype, device=new_pose.device)

        data['pose_parameters'] = new_pose

        output_file = output_dir / file.name

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)



    
if __name__ == '__main__':
    main()