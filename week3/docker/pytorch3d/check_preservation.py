"""Read-only audit of the original research environment and protected files."""
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import torch

root = Path("/home/bogger/10weekplan/week3")
results = root / "results/pytorch3d_setup_20261020"
before = json.loads((results / "environment_before.json").read_text())
after = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "packages": sorted([(d.metadata["Name"], d.version) for d in metadata.distributions()]),
    "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in before["file_sha256"]},
}
(results / "environment_after.json").write_text(json.dumps(after, indent=2) + "\n")
before_packages = [list(item) for item in before["packages"]]
after_packages = [list(item) for item in after["packages"]]
audit = {
    "package_count": len(after_packages),
    "packages_unchanged": before_packages == after_packages,
    "python_unchanged": before["python"] == after["python"],
    "torch_unchanged": before["torch"] == after["torch"],
    "cuda_unchanged": before["cuda"] == after["cuda"],
    "protected_file_count": len(after["file_sha256"]),
    "changed_files": [name for name, digest in before["file_sha256"].items()
                      if after["file_sha256"].get(name) != digest],
    "main_environment_cuda_sum": torch.tensor([1., 2., 3.], device="cuda").square().sum().item(),
}
audit["status"] = "PASS" if all(audit[key] for key in (
    "packages_unchanged", "python_unchanged", "torch_unchanged", "cuda_unchanged")) and not audit["changed_files"] else "FAIL"
(results / "preservation.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
assert audit["status"] == "PASS", "Original environment or protected files changed"
