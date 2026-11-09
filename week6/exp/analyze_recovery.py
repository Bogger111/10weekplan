import csv
import numpy as np

path = "/home/bogger/10weekplan/week6/results/v01_20261029_152830/persons.csv"

records = {}

with open(path, newline="") as f:
    for row in csv.DictReader(f):
        key = (int(row["track_id"]), int(row["frame_id"]))
        records[key] = row


events = [(3, 19, 24), (1, 40, 44), (4, 15, 16)]

for tid, t0, t1 in events:
    a = records[(tid, t0)]
    b = records[(tid, t1)]

    p0 = np.array([float(a[k]) for k in ("x", "y", "z")])
    p1 = np.array([float(b[k]) for k in ("x", "y", "z")])

    delta_xyz = p1 - p0
    distance = np.linalg.norm(delta_xyz)

    conf0 = float(a["conf"])
    conf1 = float(b["conf"])

    missing_frames = t1 - t0 - 1

    print(f"\n===== Track ID {tid} =====")
    print("From frame:", t0, "to:", t1)
    print("Delta XYZ:", delta_xyz)
    print("3D distance:", distance)
    print("Confidence:", conf0, "->", conf1)
    print("Missing frames:", missing_frames)


from pathlib import Path
from PIL import Image, ImageDraw
import csv

run = Path("/home/bogger/10weekplan/week6/results/v01_20261029_152830")

with open(run / "frames.csv", newline="") as f:
    rows = list(csv.DictReader(f))

def make_sheet(frame_ids, filename):
    w, h = 480, 300
    sheet = Image.new("RGB", (w * 3, (h + 30) * 2), "white")
    draw = ImageDraw.Draw(sheet)

    for j, t in enumerate(frame_ids):
        row = rows[t]
        image = Image.open(run / "frames" / row["frame_file"]).convert("RGB")
        image.thumbnail((w, h))

        x = (j % 3) * w
        y = (j // 3) * (h + 30)

        sheet.paste(image, (x, y))
        draw.text(
            (x + 5, y + h + 3),
            f"Frame {t} | IDs: {row['track_ids']}",
            fill="black"
        )

    sheet.save(run / filename)

make_sheet([19, 20, 21, 22, 24, 25], "recovery_id3.png")
make_sheet([39, 40, 41, 42, 43, 44], "recovery_id1.png")