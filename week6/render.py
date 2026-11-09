#!/usr/bin/env python3
"""Render original vs. smoothed Multi-HMR 2 Anny parameters as comparable MP4s.

Run in the user's `multihmr2` Python environment. This script does NOT infer HMR
or modify any PKL. It reconstructs meshes using the same Anny model, intrinsics,
and rendering settings on both sides.

Only unpickle files you created or otherwise trust.
"""

import argparse
from pathlib import Path
import pickle
import re

import cv2
import numpy as np
import torch
import anny
from multihmr2.utils.render import render_meshes


DEFAULT_ORIGINAL = Path(
    "/home/bogger/10weekplan/week6/results/"
    "blender_baseline_20261029_203921/video/anny_params"
)
DEFAULT_SMOOTHED = Path(
    "/home/bogger/10weekplan/week6/results/anny_smoothed_full"
)
DEFAULT_OUTPUT = Path(
    "/home/bogger/10weekplan/week6/results/anny_render_comparison"
)


def natural_sort_key(path: Path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", path.name)]


def load_params(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def to_cpu_float32(x):
    return torch.as_tensor(x).detach().to(device="cpu", dtype=torch.float32)


def read_camera_K(data):
    K = np.asarray(torch.as_tensor(data["K"]).detach().cpu(), dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected K (3,3); got {K.shape}")
    return K


def create_model():
    model = anny.create_fullbody_model(
        local_changes=True,
        all_phenotypes=True,
        pose_parameterization="root_relative_world",
        remove_unattached_vertices=False,
    ).to(device="cpu", dtype=torch.float32)
    model.eval()
    faces = torch.as_tensor(model.faces).detach().cpu().numpy().astype(np.int64)
    # Some legacy Anny topologies use quads; pyrender expects triangles.
    if faces.shape[1] == 4:
        faces = np.concatenate(
            [faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0
        )
    elif faces.shape[1] != 3:
        raise ValueError(f"Unsupported face topology: {faces.shape}")
    return model, faces


def reconstruct_both(model, original, smoothed):
    orig_pose = to_cpu_float32(original["pose_parameters"])
    sm_pose = to_cpu_float32(smoothed["pose_parameters"])
    orig_shape = to_cpu_float32(original["shape"])
    sm_shape = to_cpu_float32(smoothed["shape"])

    if orig_pose.shape != sm_pose.shape:
        raise ValueError("Original and smoothed pose dimensions differ")
    if orig_shape.shape != sm_shape.shape:
        raise ValueError("Original and smoothed shape dimensions differ")
    people = orig_pose.shape[0]

    # One forward pass for both versions; shared model and mesh topology.
    pose = torch.cat([orig_pose, sm_pose], dim=0)
    shape = torch.cat([orig_shape, sm_shape], dim=0)
    if shape.shape[1] != len(model.phenotype_labels):
        raise ValueError("Anny phenotype labels do not match saved shape")

    phenotypes = {
        label: shape[:, i] for i, label in enumerate(model.phenotype_labels)
    }
    local_changes = {
        label: torch.zeros(shape.shape[0], dtype=shape.dtype)
        for label in model.local_change_labels
    }
    with torch.inference_mode():
        output = model(
            pose_parameters=pose,
            phenotype_kwargs=phenotypes,
            local_changes_kwargs=local_changes,
        )
    vertices = output["vertices"].detach().cpu().numpy()
    return vertices[:people], vertices[people:], orig_pose


def camera_params(K):
    return {
        "focal": [float(K[0, 0]), float(K[1, 1])],
        "princpt": [float(K[0, 2]), float(K[1, 2])],
    }


def image_dimensions(K, width, height):
    # Assumes the principal point is approximately the image center.
    w = int(width) if width else int(round(2 * float(K[0, 2])))
    h = int(height) if height else int(round(2 * float(K[1, 2])))
    if w < 64 or h < 64 or w > 4096 or h > 4096:
        raise ValueError(
            f"Inferred image size {w}x{h} is invalid. "
            "Specify --width and --height (matching the inference image)."
        )
    return w + (w % 2), h + (h % 2)  # Even sizes for MP4 encoders


def make_writer(path, fps, size):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return writer


def labeled_pair(original_rgb, smoothed_rgb, frame_idx):
    height, width = original_rgb.shape[:2]
    # RGB -> BGR for OpenCV VideoWriter and labels.
    left = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(smoothed_rgb, cv2.COLOR_RGB2BGR)
    pair = np.concatenate([left, right], axis=1)
    # Same content and resolution in both halves; only parameters differ.
    cv2.rectangle(pair, (0, 0), (width, 39), (32, 37, 44), -1)
    cv2.rectangle(pair, (width, 0), (2 * width, 39), (32, 37, 44), -1)
    cv2.putText(pair, "ORIGINAL", (14, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pair, "SMOOTHED", (width + 14, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                2, cv2.LINE_AA)
    cv2.putText(pair, f"FRAME {frame_idx:03d}",
                (max(0, 2 * width - 188), height - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30),
                2, cv2.LINE_AA)
    return left, right, pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--smoothed", type=Path, default=DEFAULT_SMOOTHED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Output FPS (default 10, same as parameter sampling)")
    parser.add_argument("--video", type=Path, default=None,
                        help="Optional source video for rendering mesh overlays")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Source-video frame step (3 if 30fps source -> 10fps PKL)")
    parser.add_argument("--width", type=int, default=None,
                        help="Canvas width; otherwise inferred from K principal point")
    parser.add_argument("--height", type=int, default=None,
                        help="Canvas height; otherwise inferred from K principal point")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional quick test, e.g. 3")
    args = parser.parse_args()

    files = sorted(args.original.glob("*.pkl"), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No original PKLs in {args.original}")
    missing = [p.name for p in files if not (args.smoothed / p.name).is_file()]
    if missing:
        raise FileNotFoundError(f"Smoothed PKLs missing: {missing[:8]}")
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive")
        files = files[:args.max_frames]
    if args.fps <= 0 or args.frame_step < 1:
        raise ValueError("FPS and frame step must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    first = load_params(files[0])
    K_first = read_camera_K(first)
    width, height = image_dimensions(K_first, args.width, args.height)

    video = None
    if args.video is not None:
        video = cv2.VideoCapture(str(args.video))
        if not video.isOpened():
            raise FileNotFoundError(f"Cannot read source video: {args.video}")
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width += width % 2
        height += height % 2

    print(f"Frames: {len(files)}, output: {width}x{height} @ {args.fps:g} FPS")
    print("Loading Anny body model...")
    model, faces = create_model()
    print(f"Anny: {model.bone_count} bones, {len(faces)} triangles")

    original_writer = make_writer(args.out / "original.mp4", args.fps,
                                  (width, height))
    smoothed_writer = make_writer(args.out / "smoothed.mp4", args.fps,
                                  (width, height))
    comparison_writer = make_writer(args.out / "comparison.mp4", args.fps,
                                    (width * 2, height))

    colors = [(0.31, 0.70, 0.94), (0.98, 0.56, 0.29)]
    try:
        for t, file in enumerate(files):
            original = load_params(file)
            smoothed = load_params(args.smoothed / file.name)
            K = read_camera_K(original)

            if video is None:
                background = np.full((height, width, 3),
                                     (237, 241, 245), dtype=np.uint8)
            else:
                # PKL frame t is assumed to correspond to source frame t*step.
                video.set(cv2.CAP_PROP_POS_FRAMES, t * args.frame_step)
                ok, frame_bgr = video.read()
                if not ok:
                    raise RuntimeError(
                        f"Missing source-video frame {t * args.frame_step}; "
                        "check --frame-step and input video"
                    )
                background = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if background.shape[:2] != (height, width):
                    background = cv2.resize(background, (width, height))

            original_mesh, smoothed_mesh, original_pose = reconstruct_both(
                model, original, smoothed
            )
            # Fixed identity colors for this 2-person video (positive X first).
            order = torch.argsort(original_pose[:, 0, 0, 3],
                                   descending=True).cpu().tolist()
            display_colors = [colors[i % len(colors)] for i in range(len(order))]
            cam = camera_params(K)
            before_img = render_meshes(
                background.copy(), [original_mesh[i] for i in order],
                [faces] * len(order), cam, color=display_colors
            )
            after_img = render_meshes(
                background.copy(), [smoothed_mesh[i] for i in order],
                [faces] * len(order), cam, color=display_colors
            )
            left, right, pair = labeled_pair(before_img, after_img, t)
            original_writer.write(left)
            smoothed_writer.write(right)
            comparison_writer.write(pair)
            print(f"[{t + 1:03d}/{len(files):03d}] {file.name}", flush=True)
    finally:
        original_writer.release()
        smoothed_writer.release()
        comparison_writer.release()
        if video is not None:
            video.release()

    print("\nVideos saved:")
    for name in ("original.mp4", "smoothed.mp4", "comparison.mp4"):
        print("  ", args.out / name)
    print("Note: this is matched-camera visualization, not accuracy ground truth.")


if __name__ == "__main__":
    main()
