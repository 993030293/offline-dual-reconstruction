"""Shared runtime identity and atomic receipt helpers for isolated experiments."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys


REFERENCE_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"


def activate_reference(runtime, rasterizer_library=None):
    runtime = Path(runtime).resolve()
    repo = runtime / "source/gaussian-splatting"
    libraries = sorted((runtime / "runtime/rasterizer-build").glob("lib.*"))
    if len(libraries) != 1:
        raise RuntimeError("The preserved rasterizer selection is ambiguous")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError("The reference source version changed")
    library = Path(rasterizer_library).resolve() if rasterizer_library else libraries[0]
    package = library/"diff_gaussian_rasterization"
    extensions = list(package.glob("_C*.so"))
    if len(extensions) != 1 or not (package/"__init__.py").is_file():
        raise RuntimeError("The explicit rasterizer package is incomplete or ambiguous")
    if "diff_gaussian_rasterization" in sys.modules:
        raise RuntimeError("Select the rasterizer before importing a CUDA backend")
    sys.path[:0] = [str(repo), str(library)]
    return {"reference_commit": commit, "reference_source": str(repo), "rasterizer": str(library),
            "rasterizer_python": data_identity(package/"__init__.py"),
            "rasterizer_extension": data_identity(extensions[0]), "custom_rasterizer": rasterizer_library is not None}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def data_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
