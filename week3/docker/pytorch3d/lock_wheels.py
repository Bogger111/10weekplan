"""Record exact retained wheel versions and hashes after preparation."""
import hashlib
import json
import zipfile
from email.parser import Parser
from pathlib import Path

base = Path(__file__).resolve().parent
normalize = lambda name: name.lower().replace("_", "-").replace(".", "-")
required = dict(line.split("==") for line in (base / "requirements.in").read_text().splitlines()
                if line and not line.startswith("#"))
found = {}
for wheel in sorted((base / "wheelhouse").glob("*.whl")):
    with zipfile.ZipFile(wheel) as archive:
        metadata_file = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        data = Parser().parsestr(archive.read(metadata_file).decode())
    name, version = data["Name"], data["Version"]
    found[normalize(name)] = (name, version, wheel.name, hashlib.sha256(wheel.read_bytes()).hexdigest())
lines = ["# Install only on the base image pinned in Dockerfile.",
         "# PyTorch and its preinstalled dependencies are locked by the image digest."]
for name, version in sorted(required.items(), key=lambda item: normalize(item[0])):
    actual_name, actual_version, filename, digest = found[normalize(name)]
    assert actual_version == version, f"Wrong version: {filename}"
    lines.append(f"{actual_name}=={version} --hash=sha256:{digest}")
(base / "requirements.lock").write_text("\n".join(lines) + "\n")
(base / "wheelhouse.lock.json").write_text(json.dumps(list(found.values()), indent=2) + "\n")
print(f"Locked {len(required)} dependency wheels")
