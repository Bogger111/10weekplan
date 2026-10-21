"""Functional smoke test: transforms, compiled CPU/CUDA ops, mesh rasterization, SMPL.

Run through p3d.sh verify or p3d.sh verify-cpu. A requested CUDA test never
silently falls back to CPU. The model is only read from its local mount.
"""

import argparse
import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import pytorch3d
from pytorch3d import _C
from pytorch3d.ops import knn_points, sample_farthest_points, sample_points_from_meshes
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def check_transforms(device):
    aa = torch.tensor([[0.0, 0.0, math.pi / 2], [0.2, -0.3, 0.4]],
                      dtype=torch.float64, device=device)
    matrices = axis_angle_to_matrix(aa)
    expected = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                            dtype=torch.float64, device=device)
    torch.testing.assert_close(matrices[0], expected, atol=1e-10, rtol=1e-10)
    six = matrix_to_rotation_6d(matrices).detach().requires_grad_(True)
    recovered = rotation_6d_to_matrix(six)
    torch.testing.assert_close(recovered, matrices, atol=1e-10, rtol=1e-10)
    identity = torch.eye(3, dtype=torch.float64, device=device).expand_as(matrices)
    torch.testing.assert_close(recovered @ recovered.transpose(-1, -2), identity)
    torch.testing.assert_close(torch.linalg.det(recovered), torch.ones(2, dtype=torch.float64, device=device))
    weights = torch.arange(9, dtype=torch.float64, device=device).reshape(3, 3)
    (recovered * weights).sum().backward()
    assert six.grad is not None and torch.isfinite(six.grad).all() and six.grad.abs().sum() > 0
    return {"roundtrip_max_error": float((recovered - matrices).abs().max()),
            "backward_finite": True, "known_z90_rotation": True}


def check_knn_and_sampling(device):
    points = torch.tensor([[[0.1, 0., 0.], [2.8, 0., 0.]]], device=device, requires_grad=True)
    targets = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]]], device=device)
    result = knn_points(points, targets, K=1)
    expected_dist, expected_idx = torch.cdist(points.detach(), targets).square().min(dim=-1)
    torch.testing.assert_close(result.dists[..., 0], expected_dist, atol=1e-6, rtol=1e-5)
    assert torch.equal(result.idx[..., 0], expected_idx)
    assert result.dists.device == points.device
    result.dists.sum().backward()
    expected_grad = torch.tensor([[[0.2, 0., 0.], [-0.4, 0., 0.]]], device=device)
    torch.testing.assert_close(points.grad, expected_grad, atol=1e-6, rtol=1e-5)
    samples, indices = sample_farthest_points(targets, K=2, random_start_point=False)
    assert samples.shape == (1, 2, 3) and indices.shape == (1, 2)
    assert torch.equal(indices, torch.tensor([[0, 2]], device=device))
    return {"knn_squared_distances": result.dists.detach().cpu().tolist(),
            "knn_indices": result.idx.cpu().tolist(), "knn_backward": True,
            "farthest_point_indices": indices.cpu().tolist()}


def check_mesh_and_rasterizer(device):
    vertices = torch.tensor([[-0.6, -0.5, 1.], [0.6, -0.5, 1.], [0., 0.6, 1.]],
                            device=device, requires_grad=True)
    faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
    mesh = Meshes(verts=[vertices], faces=[faces])
    area = mesh.faces_areas_packed()
    torch.testing.assert_close(area, torch.tensor([0.66], device=device))
    points = sample_points_from_meshes(mesh, 32)
    assert points.shape == (1, 32, 3) and torch.isfinite(points).all()
    torch.testing.assert_close(points[..., 2], torch.ones((1, 32), device=device))
    pix_to_face, zbuf, barycentric, distances = rasterize_meshes(
        mesh, image_size=32, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    covered = pix_to_face >= 0
    coverage = int(covered.sum())
    assert 0 < coverage < 32 * 32
    assert zbuf.device == vertices.device
    assert torch.isfinite(barycentric[covered]).all()
    torch.testing.assert_close(zbuf[covered], torch.ones_like(zbuf[covered]))
    zbuf[covered].mean().backward()
    assert vertices.grad is not None and torch.isfinite(vertices.grad).all()
    torch.testing.assert_close(vertices.grad[:, 2].sum(), torch.tensor(1., device=device), atol=1e-5, rtol=1e-5)
    return {"vertices": list(mesh.verts_packed().shape), "triangle_area": float(area[0]),
            "sample_shape": list(points.shape), "covered_pixels": coverage,
            "rasterizer_backward": True}


def check_smpl(model_path, device):
    import smplx
    path = Path(model_path)
    assert path.is_file(), f"Model missing: {path}"
    read_only = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    assert read_only, "SMPL model must be mounted read-only for this validation"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    model = smplx.create(str(path), model_type="smpl", gender="neutral", num_betas=10).to(device).eval()
    with torch.no_grad():
        output = model()
        assert output.vertices.shape == (1, 6890, 3) and torch.isfinite(output.vertices).all()
        assert output.vertices.device.type == device.type
        faces = torch.as_tensor(model.faces, device=device, dtype=torch.int64)
        mesh = Meshes(verts=output.vertices, faces=faces[None])
        areas = mesh.faces_areas_packed()
        assert faces.shape == (13776, 3) and torch.isfinite(areas).all() and areas.sum() > 0
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after
    return {"vertices": list(output.vertices.shape), "faces": list(faces.shape),
            "finite": True, "model_sha256": after, "device": str(device),
            "model_mount_read_only": read_only}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--smpl-model")
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable; no fallback was used"
    device = torch.device(args.device)
    torch.manual_seed(20261020)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "pytorch3d": pytorch3d.__version__,
        "compiled_extension": _C.__file__, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if args.device == "cuda" else None,
        "checks": {},
    }
    for name, function in (("transforms", check_transforms), ("compiled_ops", check_knn_and_sampling),
                           ("mesh_and_rasterizer", check_mesh_and_rasterizer)):
        result["checks"][name] = function(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"PASS {device}: {name}", flush=True)
    if args.smpl_model:
        result["checks"]["smpl"] = check_smpl(args.smpl_model, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"PASS {device}: SMPL forward and Meshes", flush=True)
    result["status"] = "PASS"
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
