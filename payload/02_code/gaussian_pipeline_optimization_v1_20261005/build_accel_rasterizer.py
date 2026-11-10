"""Build an isolated official accelerated rasterizer without replacing an environment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime import atomic_json, data_identity


SOURCE_COMMIT = "26ce026ae9d3cfa56a103279b863a9f320c3e555"
GLM_COMMIT = "5c46b9c07008ae65cb81ab79cd677ecc1934b903"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for directory, expected in [(args.source, SOURCE_COMMIT), (args.source/"third_party/glm", GLM_COMMIT)]:
        commit = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if commit != expected:
            raise ValueError("Rasterizer or GLM pin changed")
        dirty = subprocess.check_output(["git", "-c", "core.autocrlf=true", "-C", str(directory),
            "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=dirty"], text=True).strip()
        if dirty:
            raise ValueError("Tracked source differs from the registered vendor pin")
    cuda = Path("/usr/local/cuda-12.8")
    environment = dict(os.environ)
    environment.update(CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST="12.0", MAX_JOBS="2", PATH=str(cuda/"bin")+":"+environment["PATH"])
    contract = {"source": str(args.source), "commit": SOURCE_COMMIT, "glm_commit": GLM_COMMIT,
        "primary_source": "https://github.com/graphdeco-inria/gaussian-splatting#training-speed-acceleration",
        "vendor_branch": "3dgs_accel", "license": "INRIA Gaussian-Splatting research/evaluation license; kept intact",
        "architecture": "sm_120", "cuda_home": str(cuda), "maximum_build_seconds": 900, "maximum_jobs": 2,
        "git_content_check": "command-scoped core.autocrlf=true for the Windows checkout; each submodule pin and tracked content verified separately",
        "installation_performed": False, "existing_environment_modified": False, "source_identity": data_identity(Path(__file__))}
    atomic_json(args.output/"CONTRACT.json", contract)
    command = [sys.executable, "setup.py", "build_ext", "--build-temp", str(args.output/"temp"), "--build-lib", str(args.output/"lib")]
    started = time.monotonic()
    with (args.output/"build.log").open("w") as log:
        result = subprocess.run(command, cwd=args.source, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    package = args.output/"lib/diff_gaussian_rasterization"
    if result.returncode == 0:
        shutil.copyfile(args.source/"diff_gaussian_rasterization/__init__.py", package/"__init__.py")
        shutil.copyfile(args.source/"LICENSE.md", package/"LICENSE.md")
    modules = list(package.glob("_C*.so"))
    status = "built_pending_projection_and_training_validation" if result.returncode == 0 and len(modules) == 1 else "build_failed"
    receipt = {"status": status, "returncode": result.returncode, "elapsed_seconds": time.monotonic()-started,
        "command": command, "modules": [data_identity(path) for path in modules], "counted": False}
    atomic_json(args.output/"RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)
    if status == "build_failed":
        raise RuntimeError("Isolated accelerated rasterizer build failed")


if __name__ == "__main__":
    main()
