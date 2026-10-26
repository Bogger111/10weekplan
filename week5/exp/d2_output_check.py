"""Replay the exact body layer; replay residual is not reconstruction accuracy."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--atol", type=float, default=5e-4)
    a = p.parse_args()
    run = a.run.resolve()
    m = json.loads((run / "manifest.json").read_text())
    if m["status"] != "demo_ok":
        p.error("Day 1 must contain real successful inference evidence")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    os.chdir(m["repo"])
    sys.path.insert(0, m["repo"])
    from hmr2.models import load_hmr2
    model, cfg = load_hmr2(m["checkpoint"])
    model = model.cpu().eval()
    reports = []
    with torch.no_grad():
        for name in m["captures"]:
            with np.load(run / name, allow_pickle=False) as z:
                params = {k: torch.from_numpy(z["smpl_" + k]).float()
                          for k in ("global_orient", "body_pose", "betas")}
                body = model.smpl(**params, pose2rot=False)
                row = {"capture": name, "params": {k: list(v.shape) for k, v in params.items()}}
                for key, value in (("vertices", body.vertices), ("joints", body.joints)):
                    ref = z["pred_vertices" if key == "vertices" else "pred_keypoints_3d"]
                    pred = value.cpu().numpy()
                    if pred.shape != ref.shape or not np.isfinite(pred).all():
                        raise ValueError(f"Invalid {key} shape/values")
                    row[key + "_max_abs_model_unit"] = float(np.max(np.abs(pred - ref)))
                    row[key + "_mean_l2_model_unit"] = float(np.linalg.norm(pred - ref, axis=-1).mean())
                changed = {k: v.clone() for k, v in params.items()}
                changed["betas"][:, 0] += 1.0
                variant = model.smpl(**changed, pose2rot=False)
                row["beta0_plus1_mean_vertex_displacement_model_unit"] = float(
                    torch.linalg.vector_norm(variant.vertices - body.vertices, dim=-1).mean())
                row["replay_ok"] = all(row[k + "_max_abs_model_unit"] <= a.atol
                                       for k in ("vertices", "joints"))
                row["atol_model_unit"] = a.atol
                np.savez_compressed(out / name, vertices=body.vertices.numpy(),
                    joints=body.joints.numpy(), variant_vertices=variant.vertices.numpy(),
                    faces=z["faces"], personid=z["personid"])
                reports.append(row)
    result = {"scope": "same-output replay, not GT accuracy", "device": "cpu",
              "joint_mapping": "same HMR2 SMPL wrapper; inspect cfg and smpl_wrapper.py",
              "runs": reports}
    (out / "replay.json").write_text(json.dumps(result, indent=2))
    print(out / "replay.json")
    return 0 if reports and all(r["replay_ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
