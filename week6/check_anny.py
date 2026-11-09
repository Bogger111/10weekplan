import pickle
from pathlib import Path

from collections import Counter

path = Path("/home/bogger/10weekplan/week6/results/blender_baseline_20261029_203921/video/anny_params")

files = sorted(path.glob("*.pkl"))

counts = []

for i, file in enumerate(files):
    with open(file, "rb") as f:
        data = pickle.load(f)

    pose = data["pose_parameters"]
    shape = data["shape"]

    n_people = pose.shape[0]
    counts.append(n_people)

    # Root transformation translation component
    root_pos = pose[:, 0, :3, 3]

    print(
        f"Frame {i:02d} | "
        f"People: {n_people} | "
        f"Root positions: {root_pos.tolist()}"
    )

print("\nSummary:")
print("Frames:", len(files))
print("People distribution:", Counter(counts))