"""Download pinned dependencies from PyPI, verifying PyPI's SHA256 metadata.

This prepares files only; it does not install anything in the host environment.
Pure Python source distributions are built separately in a disposable container.
"""
import argparse
import hashlib
import json
import socket
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="311")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    socket.setdefaulttimeout(30)
    metadata_dir = base / "pypi-metadata"
    metadata_dir.mkdir(exist_ok=True)
    packages = [s.strip().split("==") for s in (base / "requirements.in").read_text().splitlines()
                if s.strip() and not s.lstrip().startswith("#")]

    def open_url(url):
        for attempt in range(4):
            try:
                return urllib.request.urlopen(url)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)

    def fetch(spec):
        name, version = spec
        cache = metadata_dir / f"{name}-{version}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            for attempt in range(4):
                try:
                    with open_url(f"https://pypi.org/pypi/{name}/{version}/json") as response:
                        data = json.load(response)
                    cache.write_text(json.dumps(data))
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2)
        wheels = []
        for item in data["urls"]:
            if not item["filename"].endswith(".whl"):
                continue
            py, abi, plat = item["filename"][:-4].split("-")[-3:]
            py_ok = f"cp{args.python}" in py.split(".") or "py3" in py.split(".")
            plat_ok = plat == "any" or ("manylinux" in plat and "x86_64" in plat)
            if py_ok and plat_ok:
                wheels.append(item)
        if wheels:
            wheels.sort(key=lambda x: (f"cp{args.python}" not in x["filename"], x["filename"]))
            item = wheels[0]
            folder = base / "wheelhouse"
        else:
            item = next(x for x in data["urls"] if x["packagetype"] == "sdist")
            folder = base / "sdists"
        folder.mkdir(exist_ok=True)
        target = folder / item["filename"]
        expected = item["digests"]["sha256"]
        for attempt in range(3):
            try:
                if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                    temporary = target.with_name(target.name + ".part")
                    with open_url(item["url"]) as response, temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    assert hashlib.sha256(temporary.read_bytes()).hexdigest() == expected, f"Hash mismatch: {target.name}"
                    temporary.replace(target)
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                assert actual == expected, f"Hash mismatch: {target.name}"
                print(f"Verified {target.name} ({target.stat().st_size} bytes)", flush=True)
                return {"name": name, "version": version, "filename": target.name,
                        "url": item["url"], "sha256": actual, "folder": folder.name}
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, packages))
    (base / "downloads.lock.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
