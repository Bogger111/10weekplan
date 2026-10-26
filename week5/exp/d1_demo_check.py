"""Run the official single-image demo and capture real tensors without editing it."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shape_tree(x):
    if hasattr(x, "shape"):
        return list(x.shape)
    if isinstance(x, dict):
        return {k: shape_tree(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [shape_tree(v) for v in x]
    if hasattr(x, "vertices"):
        return {"vertices": shape_tree(x.vertices), "joints": shape_tree(x.joints)}
    return type(x).__name__


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--detector", choices=["vitdet", "regnety"], default="vitdet")
    a = p.parse_args()
    repo, image, ckpt, out = [x.resolve() for x in
                              (a.repo, a.image, a.checkpoint, a.out)]
    if not (repo / "demo.py").is_file() or not image.is_file() or not ckpt.is_file():
        p.error("Missing demo.py, input image, or checkpoint; prepare assets first")
    if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        p.error("Use a jpg/jpeg/png image")
    out.mkdir(parents=True, exist_ok=False)  # Every run has a new evidence directory.
    inp = out / "input"
    inp.mkdir()
    copied = inp / ("source" + (".jpg" if image.suffix.lower() == ".jpeg"
                               else image.suffix.lower()))
    shutil.copy2(image, copied)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                         capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                           capture_output=True, text=True)
    manifest = {"status": "started", "repo": str(repo),
                "commit": git.stdout.strip(), "git_status": dirty.stdout,
                "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
                "image": str(copied), "source_image": str(image),
                "input_sha256": sha256(copied), "python": sys.executable,
                "platform": platform.platform(), "detector": a.detector,
                "seed": 42, "captures": [], "trace": []}
    old_cwd, old_argv = Path.cwd(), sys.argv[:]
    handles = []
    code = 1
    try:
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        with open(out / "run.log", "w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import numpy as np
                import torch
                import demo
                torch.manual_seed(42)
                np.random.seed(42)
                manifest.update(torch=torch.__version__, cuda=torch.version.cuda,
                                device="cuda" if torch.cuda.is_available() else "cpu")
                original_loader = demo.load_hmr2

                def observed_loader(*args, **kwargs):
                    model, cfg = original_loader(*args, **kwargs)
                    (out / "model_config.yaml").write_text(cfg.dump())

                    def trace(name):
                        def hook(module, args, result):
                            manifest["trace"].append({"module": name,
                                "input": shape_tree(args), "output": shape_tree(result)})
                        return hook

                    for name in ("backbone", "smpl_head", "smpl"):
                        handles.append(getattr(model, name).register_forward_hook(trace(name)))

                    def capture(module, args, result):
                        batch = args[0]
                        data = {k: v.detach().cpu().numpy() for k, v in result.items()
                                if torch.is_tensor(v)}
                        for k, v in result["pred_smpl_params"].items():
                            data["smpl_" + k] = v.detach().cpu().numpy()
                        for k in ("box_center", "box_size", "img_size", "personid"):
                            data[k] = batch[k].detach().cpu().numpy()
                        focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * batch["img_size"].max()
                        full_t = demo.cam_crop_to_full(result["pred_cam"],
                            batch["box_center"].float(), batch["box_size"].float(),
                            batch["img_size"].float(), focal)
                        data.update(cam_t_full=full_t.detach().cpu().numpy(),
                                    full_focal=np.asarray(float(focal)),
                                    image_size=np.asarray(cfg.MODEL.IMAGE_SIZE),
                                    faces=np.asarray(module.smpl.faces))
                        fn = f"capture_{len(manifest['captures']):03d}.npz"
                        np.savez_compressed(out / fn, **data)
                        manifest["captures"].append(fn)
                        manifest["trace"].append({"module": "hmr2",
                            "input": shape_tree(batch), "output": shape_tree(result)})
                    handles.append(model.register_forward_hook(capture))
                    return model, cfg

                demo.load_hmr2 = observed_loader
                sys.argv = ["demo.py", "--checkpoint", str(ckpt), "--img_folder", str(inp),
                            "--out_folder", str(out / "renders"), "--save_mesh",
                            "--full_frame", "--detector", a.detector]
                manifest["argv"] = sys.argv[:]
                demo.main()
                pngs = list((out / "renders").glob("*.png"))
                objs = list((out / "renders").glob("*.obj"))
                if not manifest["captures"]:
                    manifest["status"], code = "no_detection", 2
                elif not pngs or not objs:
                    raise RuntimeError("Forward ran, but render/mesh evidence is missing")
                else:
                    manifest["status"], code = "demo_ok", 0
                manifest.update(render_count=len(pngs), mesh_count=len(objs))
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
    finally:
        for h in handles:
            h.remove()
        os.chdir(old_cwd)
        sys.argv = old_argv
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(out / "manifest.json", manifest["status"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
