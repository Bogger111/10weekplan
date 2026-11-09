import json
from multihmr2.tracker import FeatPelvisTracker
import argparse
import csv
from pathlib import Path
import torch
from PIL import Image, ImageDraw

from multihmr2.api import init_hmr_session, infer_video
from multihmr2.utils.image import nonsquare_and_resize_image_without_K


# =========================
# 1. Arguments
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

out_dir = Path(args.out)
frames_dir = out_dir / "frames"
overlay_dir = out_dir / "track_overlays"

if frames_dir.exists() and any(frames_dir.iterdir()):
    raise RuntimeError(f"Frames directory is not empty: {frames_dir}")

frames_dir.mkdir(parents=True, exist_ok=True)
overlay_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Model inference
# =========================

session = init_hmr_session("checkpoints/multihmr2.pt")

preds = infer_video(
    session,
    video_path=args.video,
    tmp_dir=str(frames_dir),
    conf_thresh=0.4,
    dist_thresh_nms=0.25,
)

# ==========================================
# E4.5: Tracking Ablation
# ==========================================

modes = {
    "combined": "wsum_1.0_8.0",
    "feature_only": "wsum_1.0_0.0",
    "pelvis_only": "wsum_0.0_1.0",
}

trackers = {
    name: FeatPelvisTracker(combine=mode)
    for name, mode in modes.items()
}

results = {name: [] for name in modes}

# Replay exactly the same detections
for t, pred in enumerate(preds):
    persons = pred.persons

    if len(pred) == 0:
        # Match the official infer_video empty-frame behavior
        for name in modes:
            results[name].append([])
        continue

    features = persons.trackfeat.clone().float()
    xyz = persons.transl_pelvis.clone().float()

    # Match official API expression and conversion order
    ijn = torch.cat(
        [
            persons.j2d[:, 0] / session.img_size,
            torch.log(1 / xyz[:, 2:3].clamp_min(1e-5)),
        ],
        dim=-1,
    ).clone().float()

    ori = persons.bone_poses[:, 0, :3, :3].clone().float()

    for name, tracker in trackers.items():
        assigned = tracker.track_next_frame(
            t,
            features.clone(),
            xyz.clone(),
            ijn.clone(),
            ori.clone(),
        )
        results[name].append(assigned.cpu().tolist())


# ==========================================
# Baseline replay verification
# ==========================================

reference = [
    pred.persons.track_id.flatten().cpu().tolist()
    for pred in preds
]

mismatches = [
    t for t in range(len(preds))
    if results["combined"][t] != reference[t]
]

print("\n===== Ablation Summary =====")
print("Combined replay mismatches:", mismatches)

for name, frames in results.items():
    unique_ids = {
        tid for frame_ids in frames for tid in frame_ids
    }
    print(f"{name}: Unique IDs = {len(unique_ids)}")

print("\n===== Critical Frames =====")

for t in [19, 20, 23, 24, 40, 41, 43, 44]:
    print(f"\nFrame {t:03d}")
    for name in modes:
        print(f"  {name}: {results[name][t]}")


# Save all ablation results
with open(out_dir / "ablation.json", "w") as f:
    json.dump(
        {
            "modes": modes,
            "reference": reference,
            "results": results,
            "combined_mismatches": mismatches,
        },
        f,
        indent=2,
    )

frame_files = sorted(frames_dir.glob("*.png"))

assert len(preds) > 0, "No video frames were processed"
assert len(frame_files) == len(preds), (
    f"Frames: {len(frame_files)}, predictions: {len(preds)}"
)


# =========================
# 3. Collect tracking data
# =========================

frame_rows = []
person_rows = []

for t, pred in enumerate(preds):
    persons = pred.persons
    n = len(pred)

    assert persons.track_id is not None

    ids = persons.track_id.flatten().cpu().tolist()

    assert len(ids) == n
    assert len(ids) == len(set(ids)), (
        f"Duplicate track IDs at frame {t}: {ids}"
    )

    frame_rows.append({
        "frame_id": t,
        "frame_file": frame_files[t].name,
        "num_persons": n,
        "track_ids": str(ids),
    })

    if n > 0:
        conf = persons.conf.flatten().float().cpu()
        pelvis = persons.transl_pelvis.float().cpu()

        # 2D pelvis in model image coordinates
        uv = persons.j2d[:, 0, :].float().cpu()

        assert uv.shape == (n, 2)
        assert pelvis.shape == (n, 3)

        for i in range(n):
            x, y, z = pelvis[i].tolist()
            u, v = uv[i].tolist()

            person_rows.append({
                "frame_id": t,
                "person_idx": i,
                "track_id": int(ids[i]),
                "conf": float(conf[i].item()),
                "x": x,
                "y": y,
                "z": z,
                "u": u,
                "v": v,
            })

    print(f"Frame {t:03d}: N={n}, IDs={ids}")


# =========================
# 4. Save CSV
# =========================

def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


save_csv(
    out_dir / "frames.csv",
    frame_rows,
    ["frame_id", "frame_file", "num_persons", "track_ids"],
)

save_csv(
    out_dir / "persons.csv",
    person_rows,
    [
        "frame_id",
        "person_idx",
        "track_id",
        "conf",
        "x", "y", "z",
        "u", "v",
    ],
)


# =========================
# 5. Tracking summary
# =========================

unique_ids = sorted(set(r["track_id"] for r in person_rows))

print("\n===== Tracking Summary =====")
print("Total frames:", len(preds))
print("Total detections:", len(person_rows))
print("Unique IDs:", len(unique_ids))
print("Track IDs:", unique_ids)

# Compare with the previous V01 experiment.
if len(preds) == 45:
    if len(person_rows) != 203 or len(unique_ids) != 5:
        print(
            "WARNING: Tracking counts differ from previous V01 "
            "(203 detections, 5 IDs)."
        )


# =========================
# 6. Track ID overlays
# =========================

COLORS = [
    "#FF3838",  # ID 0: red
    "#00D9FF",  # ID 1: cyan
    "#39E75F",  # ID 2: green
    "#FFD000",  # ID 3: yellow
    "#DA70D6",  # ID 4: purple
    "#FF8C00",
]

events = {
    "id3_recovery": [19, 20, 21, 22, 23, 24],
    "id1_recovery": [39, 40, 41, 42, 43, 44],
}


def draw_frame(t):
    pred = preds[t]
    persons = pred.persons

    # Same geometry preprocessing as official file-path inference.
    img = Image.open(frame_files[t]).convert("RGB")
    img = nonsquare_and_resize_image_without_K(
        img, 768, 16
    )

    draw = ImageDraw.Draw(img)

    if len(pred) == 0:
        return img

    xy = persons.j2d[:, 0, :].float().cpu().tolist()
    ids = persons.track_id.flatten().cpu().tolist()

    assert len(xy) == len(ids)

    for (u, v), tid in zip(xy, ids):
        color = COLORS[int(tid) % len(COLORS)]

        if not (0 <= u < img.width and 0 <= v < img.height):
            print(
                f"WARNING: Frame {t}, ID {tid}, "
                f"pelvis outside image: ({u:.1f}, {v:.1f})"
            )
            continue

        r = 6

        draw.ellipse(
            (u - r, v - r, u + r, v + r),
            fill=color,
            outline="black",
            width=2,
        )

        # Avoid placing the label outside the image.
        tx = max(0, min(int(u + 9), img.width - 65))
        ty = max(0, min(int(v - 17), img.height - 20))

        draw.text(
            (tx, ty),
            f"ID {tid}",
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )

    return img


# =========================
# 7. Recovery comparison sheets
# =========================

for name, frame_ids in events.items():
    if max(frame_ids) >= len(preds):
        print(f"Skipping {name}: insufficient frames")
        continue

    images = [draw_frame(t) for t in frame_ids]

    w, h = images[0].size
    caption_height = 34

    sheet = Image.new(
        "RGB",
        (3 * w, 2 * (h + caption_height)),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for i, (t, img) in enumerate(zip(frame_ids, images)):
        x = (i % 3) * w
        y = (i // 3) * (h + caption_height)

        sheet.paste(img, (x, y))

        ids = preds[t].persons.track_id.flatten().tolist()

        draw.text(
            (x + 8, y + h + 7),
            f"Frame {t:03d} | IDs: {ids}",
            fill="black",
        )

    target = overlay_dir / f"{name}.png"
    sheet.save(target)

    print("Saved:", target)


print("\n===== Finished =====")
print("Results:", out_dir.resolve())
