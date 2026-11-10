"""Capture installed native source and extension identities without changing the environment."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    source = Path("/opt/e26_thin_surface_v1/pgsr")
    output = args.root / "implementation_receipt"
    output.mkdir(exist_ok=False)
    paths = [source / name for name in ("train.py", "scene/cameras.py", "scene/gaussian_model.py",
        "scene/__init__.py", "scene/dataset_readers.py", "utils/camera_utils.py", "utils/graphics_utils.py",
        "gaussian_renderer/__init__.py", "submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu",
        "submodules/diff-plane-rasterization/cuda_rasterizer/backward.cu",
        "submodules/diff-plane-rasterization/cuda_rasterizer/rasterizer_impl.cu")]
    records = []
    for path in paths:
        target = output / "PGSR" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records.append(identity(path))
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    diff = subprocess.check_output(["git", "-C", str(source), "diff", "--", "gaussian_renderer/__init__.py"], text=True)
    (output / "PGSR_LOCAL_RENDERER_DIFF.patch").write_text(diff)
    import diff_plane_rasterization
    extension = next(Path(diff_plane_rasterization.__file__).parent.glob("_C*.so"))
    result = {"status": "installed_implementation_captured_read_only", "captured_unix": time.time(),
        "PGSR_head": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "PGSR_tracked_worktree_status": status, "PGSR_sources": records, "plane_extension": identity(extension),
        "capture_time_boundary": "after_PGSR_training_during_surface_review_not_claimed_as_prelaunch_capture",
        "renderer_already_exposed_native_alpha_before_this_run": True,
        "preexisting_local_renderer_formula_change_observed": False,
        "environment_installed_or_migrated": False, "runtime_LLM_required": False}
    (output / "RESULT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result["status"])


if __name__ == "__main__":
    main()
