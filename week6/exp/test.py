#!/usr/bin/env python3
"""Simple Multi-HMR 2 image/video baseline with Blender-importable GLB output.

Run inside the existing WSL multihmr2 environment. Image/video are optional,
but at least one must be supplied. No offscreen rendering is required.

Example:
    python test_multihmr2_blender.py \
        --image ~/10weekplan/week6/data/blender_test/two_people.jpg \
        --video ~/10weekplan/week6/data/blender_test/two_people.mp4
"""

import argparse
import csv
import datetime as dt
import shutil
import subprocess
import tempfile
from pathlib import Path

# Important for this user's WSL/PyOpenGL setup: import the Torch stack before
# the project modules (observed to avoid a native import-order crash).
import torch
try:
    import triton  # noqa: F401
    import torch._dynamo  # noqa: F401
except ImportError:
    pass

import trimesh
from multihmr2 import (
    init_hmr_session,
    infer_image,
    infer_video,
    save_results_image,
    save_results_video,
    render_results_image,
)

REPO = Path.home() / "research-repos/multi-hmr2"
DATA = Path.home() / "10weekplan/week6/data/blender_test"
RESULTS = Path.home() / "10weekplan/week6/results"
PALETTE = [
    (59, 130, 246, 255),
    (239, 68, 68, 255),
    (16, 185, 129, 255),
    (245, 158, 11, 255),
    (168, 85, 247, 255),
    (236, 72, 153, 255),
]


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def preprocess_video(source: Path, target: Path, *, fps: int,
                     seconds: float, max_frames: int, width: int) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is missing: sudo apt install ffmpeg")
    # CRITICAL: --framerate in the official renderer does NOT cap inference frames.
    # Actually create a short input MP4 BEFORE calling infer_video().
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-t", str(seconds),
        "-vf", f"fps={fps},scale={width}:-2",
        "-frames:v", str(max_frames),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(target),
    ])
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"No output video was generated: {target}")
    if shutil.which("ffprobe"):
        result = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1",
            str(target),
        ], check=True, text=True, capture_output=True)
        print(f"[video] frames to infer: {result.stdout.strip()} (cap {max_frames})")


def image_from_video(video: Path, target: Path) -> None:
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-frames:v", "1", str(target),
    ])


def export_combined_glb(pred, session, target: Path, *, lowres: bool,
                        frame_label: str) -> int:
    """Combine each visible person's camera-space mesh into ONE Blender GLB."""
    count = len(pred)
    if count == 0:
        print(f"[skip] {frame_label}: no predicted person")
        return 0
    model = (session.model.full_body_decoder.lowres_body_model if lowres
             else session.model.full_body_decoder.body_model)
    faces = model.faces.detach().cpu().numpy()
    scene = trimesh.Scene()
    track_ids = None
    if getattr(pred.persons, "track_id", None) is not None:
        track_ids = pred.persons.track_id.flatten().tolist()
    for idx in range(count):
        verts = pred.persons.v3d[idx].detach().cpu().numpy().reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        track_id = int(track_ids[idx]) if track_ids is not None else idx
        mesh.visual.vertex_colors = PALETTE[track_id % len(PALETTE)]
        scene.add_geometry(mesh, node_name=f"person_{idx:03d}_track_{track_id}")
    target.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(target), file_type="glb")
    print(f"[GLB] {target} ({count} people)")
    return count


def write_track_manifest(preds, output_csv: Path) -> None:
    """Important: the _000/_001 in the official GLB names are LOCAL indices."""
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_1_based", "local_person_index", "track_id", "glb_filename"])
        for frame_no, pred in enumerate(preds, 1):
            ids = pred.persons.track_id.flatten().tolist()
            for local_index, tid in enumerate(ids):
                writer.writerow([
                    frame_no, local_index, int(tid),
                    f"{frame_no:06d}_{local_index:03d}.glb",
                ])
    print(f"[tracking] {output_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, help="A simple full-body photograph (jpg/png)")
    p.add_argument("--video", type=Path, help="A simple multi-person video (mp4)")
    p.add_argument("--checkpoint", type=Path, default=REPO / "checkpoints/multihmr2.pt")
    p.add_argument("--out", type=Path, help="New/empty output directory")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--conf-thresh", type=float, default=0.4)
    p.add_argument("--lowres", action="store_true", help="613-vertex faster export, less detail")
    p.add_argument("--render-image", action="store_true", help="Optional 2D image overlay (needs OpenGL)")
    args = p.parse_args()

    if args.image is None and args.video is None:
        p.error("Provide --image and/or --video")
    for label, path in [("image", args.image), ("video", args.video)]:
        if path is not None and not path.expanduser().is_file():
            p.error(f"--{label} not found: {path}")
    if not args.checkpoint.expanduser().is_file():
        p.error(f"checkpoint not found: {args.checkpoint}")
    if args.fps <= 0 or args.max_frames <= 0 or args.seconds <= 0 or args.width <= 0:
        p.error("fps, max-frames, seconds, width must all be > 0")

    tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.out or (RESULTS / f"blender_baseline_{tag}")).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        p.error(f"output directory is not empty (prevents stale files): {out}")
    out.mkdir(parents=True, exist_ok=True)
    blender_dir = out / "blender"
    blender_dir.mkdir()
    prepared = None
    if args.video:
        prepared = out / f"video_prepared_{args.fps}fps.mp4"
        preprocess_video(args.video.expanduser(), prepared, fps=args.fps,
                         seconds=args.seconds, max_frames=args.max_frames,
                         width=args.width)
    image = args.image.expanduser() if args.image else None
    if image is None and prepared is not None:
        image = out / "image_from_video.jpg"
        image_from_video(prepared, image)
        print("[image] Using the FIRST frame from the same video as image baseline")

    print("[model] Loading Multi-HMR 2 once (compile_model=False)...", flush=True)
    session = init_hmr_session(str(args.checkpoint.expanduser()), compile_model=False)

    if image is not None:
        print(f"[image] {image}", flush=True)
        pred = infer_image(session, str(image), conf_thresh=args.conf_thresh,
                           lowres=args.lowres)
        image_out = out / "image"
        image_out.mkdir()
        save_results_image(session, pred, str(image_out),
                           save_mesh=True, save_anny_params=True,
                           lowres=args.lowres)
        export_combined_glb(pred, session, blender_dir / "image_all_people.glb",
                            lowres=args.lowres, frame_label="image")
        if args.render_image:
            render_results_image(session, pred, str(image), str(image_out),
                                 lowres=args.lowres)

    if prepared is not None:
        # Use a fresh temporary directory: no stale frames from earlier tests.
        with tempfile.TemporaryDirectory(prefix="multihmr2_frames_") as tmp:
            print(f"[video] Inferring only the SHORTENED clip: {prepared}", flush=True)
            predictions = infer_video(session, str(prepared), tmp_dir=tmp,
                                      conf_thresh=args.conf_thresh, lowres=args.lowres)
            video_out = out / "video"
            video_out.mkdir()
            save_results_video(session, predictions, str(video_out),
                               save_mesh=True, save_anny_params=True,
                               lowres=args.lowres)
            write_track_manifest(predictions, out / "track_id_manifest.csv")
            if predictions:
                for i in sorted(set([0, len(predictions) // 2, len(predictions) - 1])):
                    export_combined_glb(
                        predictions[i], session,
                        blender_dir / f"video_frame_{i+1:06d}_all.glb",
                        lowres=args.lowres, frame_label=f"video frame {i+1}",
                    )
            print(f"[video] processed {len(predictions)} frames, not the original video's frame count")

    print("\nFinished. In Blender: File > Import > glTF 2.0 (.glb/.gltf)")
    print(f"Easy-to-open combined 3D scenes: {blender_dir}")
    print(f"Individual image GLBs:          {out / 'image/meshes'}")
    print(f"Individual video-frame GLBs:    {out / 'video/meshes'}")
    print("Video GLBs are per-frame static meshes; NOT a single animated rig.")


if __name__ == "__main__":
    main()
