import torch
import csv
from collections import defaultdict

tracks = defaultdict(list)

with open('/home/bogger/10weekplan/week6/results/v01_20261029_152830/persons.csv', newline='') as f:
    reader = csv.DictReader(f)

    for row in reader:
        track_ids = int(row['track_id'])
        frame_ids = int(row['frame_id'])

        tracks[track_ids].append(frame_ids)

    for track_id, frames  in sorted(tracks.items()):
        frames = sorted(frames)

        length = len(frames)

        span = abs(frames[0] - frames[-1]) + 1

        gaps = span - length

        print(track_id, length, span, gaps)