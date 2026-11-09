from pathlib import Path
import pickle
import torch

original_dir = Path('/home/bogger/10weekplan/week6/results/blender_baseline_20261029_203921/video/anny_params')
smoothed_dir = Path('/home/bogger/10weekplan/week6/results/anny_smoothed')

original_files = sorted(original_dir.glob("*.pkl"))

for original_file in original_files:
    smoothed_file = smoothed_dir / original_file.name

    with open(original_file, 'rb') as f:
        original = pickle.load(f)

    with open(smoothed_file, 'rb') as f:
        filtered = pickle.load(f)

    assert torch.equal(original['K'], filtered['K'])
    assert torch.equal(original['shape'], filtered['shape'])

    pose_before = original['pose_parameters']
    pose_after = filtered['pose_parameters']

    mask = torch.zeros_like(pose_before, dtype=torch.bool)
    mask[:, 0, :3, 3] = True

    assert torch.equal(
        pose_before[~mask],
        pose_after[~mask]
    ), f"Validation failed: {original_file.name}"

    assert torch.isfinite(pose_after).all()

print(f"All {len(original_files)} frames passed!")