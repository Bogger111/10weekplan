import csv
import numpy as np

path = "/home/bogger/10weekplan/week6/results/v01_overlay_20261029_162026/persons.csv"

records = {}

with open(path, newline="") as f:
    for row in csv.DictReader(f):
        key = (int(row["frame_id"]), int(row["track_id"]))
        records[key] = row

cases = [
    (19, 3, 0),
    (24, 3, 0),
    (40, 1, 2),
    (44, 1, 2),
]

for frame, id_a, id_b in cases:
    a = records[(frame, id_a)]
    b = records[(frame, id_b)]

    uv_a = np.array([float(a[k]) for k in ("u", "v")])
    uv_b = np.array([float(b[k]) for k in ("u", "v")])

    xyz_a = np.array([float(a[k]) for k in ("x", "y", "z")])
    xyz_b = np.array([float(b[k]) for k in ("x", "y", "z")])

    d2 = np.linalg.norm(uv_a - uv_b)
    d3 = np.linalg.norm(xyz_a - xyz_b)

    print(f"\nFrame {frame} | ID {id_a} vs {id_b}")
    print(f"2D pelvis distance: {d2:.3f} px")
    print(f"3D pelvis distance: {d3:.3f}")
    print(f"Depth difference: {abs(xyz_a[2] - xyz_b[2]):.3f}")
    print(f"Confidence: {float(a['conf']):.3f}, {float(b['conf']):.3f}")