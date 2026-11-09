"""Multi-HMR 2 official-vs-smoothed overlay comparison (person-anchor fix).

Run inside the working multihmr2 environment:
    python comparison.py --max-frames 3
    python comparison.py

Requires the original video, model checkpoint, ffmpeg, SciPy and the
Multi-HMR 2 rendering dependencies. Never uses previously exported .pkl
as the source of the original pose (their root pose was overwritten on export).
"""

# IMPORTANT: preserve the import order that fixed the local SIGSEGV.
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

import torch
import triton  # noqa: F401 - intentional pre-import
import torch._dynamo  # noqa: F401 - intentional pre-import

from pathlib import Path
import argparse
import subprocess
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from multihmr2 import init_hmr_session, infer_video
from multihmr2.datasets.itw_image import preprocess_image
from multihmr2.utils import denormalize_rgb, render_meshes


VIDEO = Path("/home/bogger/10weekplan/week6/results/blender_baseline_20261029_203921/video_prepared_10fps.mp4")
CHECKPOINT = Path("/home/bogger/research-repos/multi-hmr2/checkpoints/multihmr2.pt")
OUTPUT = Path("/home/bogger/10weekplan/week6/results/comparison_official_fixed")

FPS = 10
ROOT_CUTOFF = 2.0
SHAPE_CUTOFF = 1.0
ROT_ALPHA = 0.3

# RGB floats in [0, 1], stable by track ID.
COLORS = [(0.16, 0.70, 0.95), (1.0, 0.58, 0.20), (0.35, 0.85, 0.55)]


def filter_numeric(x, cutoff):
    if len(x) < 12:
        return x.copy()  # avoid unreliable short-sequence filtfilt padding
    sos = butter(2, cutoff, fs=FPS, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x, axis=0).copy()


def interp_rotation(a, b, alpha):
    relative = a.inv() * b
    return a * Rotation.from_rotvec(alpha * relative.as_rotvec())


def filter_rotation(seq):
    if len(seq) < 2:
        return seq.copy()
    r = Rotation.from_matrix(seq)
    fw = [r[0]]
    for t in range(1, len(seq)):
        fw.append(interp_rotation(fw[-1], r[t], ROT_ALPHA))
    bw = [None] * len(seq)
    bw[-1] = fw[-1]
    for t in range(len(seq) - 2, -1, -1):
        bw[t] = interp_rotation(bw[t + 1], fw[t], ROT_ALPHA)
    return np.stack([v.as_matrix() for v in bw])


def consecutive_groups(items):
    """Separate tracks at gaps; never filter across missing observations."""
    if not items:
        return
    start = 0
    for i in range(1, len(items) + 1):
        if i == len(items) or items[i][0] != items[i - 1][0] + 1:
            yield items[start:i]
            start = i


def anny_forward(body, pose, shape, device, parameterization):
    """Use the *actual decoder body model*, including its 6 active phenotypes and LBS."""
    pose = pose.to(device=device, dtype=torch.float32)
    shape = shape.to(device=device, dtype=torch.float32)
    phenotypes = {
        name: shape[:, j]
        for j, name in enumerate(body.phenotype_labels)
    }
    local_changes = {
        name: torch.zeros(shape.shape[0], dtype=shape.dtype, device=device)
        for name in body.local_change_labels
    }
    with torch.inference_mode():
        return body(
            pose_parameters=pose,
            phenotype_kwargs=phenotypes,
            local_changes_kwargs=local_changes,
            pose_parameterization=parameterization,
        )


def align_to_pelvis(body_output, pelvis_camera):
    """Match official decoder: V_camera = V_body - B0_body + pelvis_camera."""
    vertices = body_output["vertices"]
    model_pelvis = body_output["bone_poses"][:, 0, :3, 3]
    return vertices - model_pelvis[:, None, :] + pelvis_camera[:, None, :]


def render_one(frame_path, vertices, faces, K, ids):
    # Exactly the official preprocessing and camera-intrinsics convention.
    x = preprocess_image(str(frame_path))
    image = denormalize_rgb(x[0].cpu().numpy())
    image = np.ascontiguousarray(image)
    if not len(vertices):
        return image
    k = K.detach().cpu().numpy()
    camera = {
        "focal": np.array([k[0, 0], k[1, 1]]),
        "princpt": np.array([k[0, 2], k[1, 2]]),
    }
    colors = [COLORS[int(tid) % len(COLORS)] for tid in ids]
    return render_meshes(
        image.copy(),
        [v.reshape(-1, 3) for v in vertices],
        [faces] * len(vertices),
        camera,
        color=colors,
    )


def render_frames(preds, frames, body, device, local_pose, sm_pose, sm_shape, sm_anchor, limit, mode, center_idx):
    vis_dir = OUTPUT / f"comparison_frames_{mode}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for old in vis_dir.glob("*.png"):
        old.unlink()
    faces = body.faces.detach().cpu().numpy() if isinstance(body.faces, torch.Tensor) else np.asarray(body.faces)

    for t in range(limit):
        p = preds[t]
        ids = p.persons.track_id.detach().cpu().tolist()
        original_v = p.persons.v3d.detach().cpu().numpy()  # OFFICIAL vertices, unchanged

        if len(p):
            # Align *everything* to the trained person-center vertex (5054 in
            # the official full-resolution decoder), NOT to the pose-dependent pelvis.
            official_anchor = original_v[:, center_idx, :]  # (N, 3)
            smooth_anchor_np = sm_anchor[t].detach().cpu().numpy()
            delta = (smooth_anchor_np - official_anchor)[:, None, :]

            if mode in ("rotation", "shape", "all"):
                # Deformation is expressed relative to the SAME mesh anchor;
                # any differences in Anny's absolute bone locations cancel out.
                out_base = anny_forward(
                    body, local_pose[t], p.persons.shape.detach().cpu(),
                    device, "local-bone")
                out_smoothed = anny_forward(
                    body, sm_pose[t], sm_shape[t], device, "local-bone")
                v0 = out_base["vertices"]
                v1 = out_smoothed["vertices"]
                rel0 = v0 - v0[:, center_idx:center_idx + 1, :]
                rel1 = v1 - v1[:, center_idx:center_idx + 1, :]
                deformation = (rel1 - rel0).detach().cpu().numpy()
                delta = delta + deformation
            # In anchor-only mode, NO Anny forward pass is involved in
            # generating the smooth video; it is pure official-v3d translation.
            smooth_v = original_v + delta
        else:
            smooth_v = original_v

        left = render_one(frames[t], original_v, faces, p.K, ids)
        right = render_one(frames[t], smooth_v, faces, p.K, ids)
        h, w = left.shape[:2]
        combined = Image.new("RGB", (2 * w, h + 40), (22, 25, 31))
        combined.paste(Image.fromarray(left), (0, 40))
        combined.paste(Image.fromarray(right), (w, 40))
        draw = ImageDraw.Draw(combined)
        draw.text((12, 12), "ORIGINAL / OFFICIAL v3d", fill="white")
        draw.text((w + 12, 12), f"SMOOTHED / {mode.upper()} ANCHOR-ALIGNED", fill="white")
        combined.save(vis_dir / f"{t + 1:06d}.png")
        print(f"Rendered {t + 1}/{limit}", flush=True)

    video_path = OUTPUT / f"comparison_{mode}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(FPS),
        "-start_number", "1", "-i", str(vis_dir / "%06d.png"),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags", "+faststart", str(video_path)
    ], check=True)
    print("Saved:", video_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Render just first N frames; inference and smoothing still use the whole video")
    parser.add_argument("--mode", choices=["off", "anchor", "rotation", "shape", "all"],
                        default="anchor", help="Ablation: isolate the smoothing component")
    parser.add_argument("--tracking", choices=["x", "official"], default="x",
                        help="x: stable 2-person left/right assignment for THIS video")
    args = parser.parse_args()
    print(f"MODE={args.mode} TRACKING={args.tracking}", flush=True)
    if not VIDEO.is_file():
        raise FileNotFoundError(VIDEO)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames_dir = OUTPUT / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    print("Official Multi-HMR 2 inference (with tracking)...", flush=True)
    session = init_hmr_session(str(CHECKPOINT), compile_model=False)
    preds = infer_video(session, str(VIDEO), tmp_dir=str(frames_dir))
    frames = sorted(frames_dir.glob("*.png"))
    if not preds or len(preds) != len(frames):
        raise RuntimeError(f"Inference/frame count mismatch: {len(preds)} vs {len(frames)}")
    print(f"Inferred {len(preds)} frames", flush=True)

    body = session.model.full_body_decoder.body_model
    device = session.device
    # Decoder already sets LBS. Explicitly keep the same method.
    body.set_skinning_method("lbs")

    center_idx = int(session.model.full_body_decoder.person_center_idx)
    print(f"Official person-center vertex index: {center_idx}", flush=True)

    # Critical: these are the untouched bone_poses in the live predictions.
    # The exported PKLs overwrite bone_poses[0] translation and CANNOT be
    # reconstructed by simply passing that PKL through Anny in 'world' mode.
    local_pose, base_shape, pelvis, anchor = [], [], [], []
    for t, pred in enumerate(preds):
        n = len(pred)
        if n == 0:
            local_pose.append(torch.empty((0, body.bone_count, 4, 4)))
            base_shape.append(torch.empty((0, 11)))
            pelvis.append(torch.empty((0, 3)))
            anchor.append(torch.empty((0, 3)))
            continue
        shape = pred.persons.shape.detach().cpu().float()
        world_poses = pred.persons.bone_poses.detach().cpu().float()
        cam_pelvis = pred.persons.transl_pelvis.detach().cpu().reshape(n, 3).float()
        # Convert the ORIGINAL world transforms to genuine local-bone parameters.
        world_result = anny_forward(body, world_poses, shape, device, "world")
        local = body.get_pose_parameterization(
            world_result, pose_parameterization="local-bone"
        ).detach()

        # Mandatory one-frame geometry fidelity test before any smoothing.
        if not any(p.shape[0] for p in local_pose):
            test_result = anny_forward(body, local, shape, device, "local-bone")
            reconstructed = align_to_pelvis(test_result, cam_pelvis.to(device))
            official = pred.persons.v3d.detach().to(device)
            if reconstructed.shape != official.shape:
                raise RuntimeError(f"Vertex shape mismatch: {reconstructed.shape} != {official.shape}")
            delta = torch.linalg.vector_norm(reconstructed - official, dim=-1)
            mean_error = delta.mean().item()
            max_error = delta.max().item()
            print(f"Original geometry match: mean={mean_error:.6f}m max={max_error:.6f}m", flush=True)
            # Minor round-trip residuals are expected. Since the rendered
            # smoothed mesh is a delta on top of official v3d, don't reject
            # small residuals. Reject only gross geometry mismatches.
            if mean_error > 0.02 or max_error > 0.10:
                raise RuntimeError(
                    "Large reconstruction mismatch. Check Anny pose conversion "
                    "and body-model configuration before rendering."
                )
            elif mean_error > 0.005:
                print(
                    "NOTE: Anny round-trip residual is above 5 mm; "
                    "using delta-transfer onto official v3d for visual comparison.",
                    flush=True,
                )

        local_pose.append(local.cpu().clone())
        base_shape.append(shape)
        pelvis.append(cam_pelvis)
        anchor.append(pred.persons.v3d[:, center_idx, :].detach().cpu().float().clone())

    sm_pose = [v.clone() for v in local_pose]
    sm_shape = [v.clone() for v in base_shape]
    sm_anchor = [v.clone() for v in anchor]

    # Use X-order only for this known 2-person, non-crossing test clip.
    tracks = defaultdict(list)
    for t, pred in enumerate(preds):
        if args.tracking == "x":
            if len(pred) != 2:
                raise RuntimeError(f"X-order test requires exactly 2 persons: frame {t}, got {len(pred)}")
            person_x = pred.persons.v3d[:, center_idx, 0].detach().cpu()
            order = torch.argsort(person_x, descending=True).tolist()
            for stable_id, pidx in enumerate(order):
                tracks[stable_id].append((t, pidx))
        else:
            for pidx, tid in enumerate(pred.persons.track_id.cpu().tolist()):
                tracks[int(tid)].append((t, pidx))

    for track_id, observations in tracks.items():
        for group in consecutive_groups(observations):
            anchors = np.stack([anchor[t][p].numpy() for t, p in group])
            smooth_anchors = filter_numeric(anchors, ROOT_CUTOFF) if args.mode in ("anchor", "all") else anchors.copy()
            if len(anchors) >= 3:
                j0 = float(np.linalg.norm(np.diff(anchors, n=2, axis=0), axis=-1).mean())
                j1 = float(np.linalg.norm(np.diff(smooth_anchors, n=2, axis=0), axis=-1).mean())
                print(f"Track {track_id} anchor second-difference: {j0:.5f} -> {j1:.5f} m/frame^2", flush=True)

            if args.mode in ("shape", "all"):
                shapes = np.stack([base_shape[t][p].numpy() for t, p in group])
                smooth_shapes = np.clip(filter_numeric(shapes, SHAPE_CUTOFF), 0.0, 1.0)
            else:
                smooth_shapes = None

            if args.mode in ("rotation", "all"):
                rotations = np.stack([local_pose[t][p, :, :3, :3].numpy() for t, p in group])
                smooth_rotations = np.empty_like(rotations)
                for joint in range(rotations.shape[1]):
                    smooth_rotations[:, joint] = filter_rotation(rotations[:, joint])
            else:
                smooth_rotations = None

            for k, (t, p) in enumerate(group):
                sm_anchor[t][p] = torch.as_tensor(smooth_anchors[k].copy(), dtype=sm_anchor[t].dtype)
                if smooth_shapes is not None:
                    sm_shape[t][p] = torch.as_tensor(smooth_shapes[k].copy(), dtype=sm_shape[t].dtype)
                if smooth_rotations is not None:
                    sm_pose[t][p, :, :3, :3] = torch.as_tensor(smooth_rotations[k].copy(), dtype=sm_pose[t].dtype)
        print(f"Track {track_id}: {len(observations)} observations", flush=True)

    limit = len(preds) if args.max_frames is None else min(args.max_frames, len(preds))
    render_frames(preds, frames, body, device, local_pose, sm_pose, sm_shape, sm_anchor, limit, args.mode, center_idx)


if __name__ == "__main__":
    main()
