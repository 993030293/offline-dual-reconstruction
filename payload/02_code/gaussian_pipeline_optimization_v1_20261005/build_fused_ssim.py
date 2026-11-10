"""Build the reference-pinned optional SSIM extension without installing packages."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime import atomic_json, data_identity


PINNED_COMMIT = "1272e21a282342e89537159e4bad508b19b34157"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("The SSIM source differs from the preserved INRIA submodule pin")
    cuda = Path("/usr/local/cuda-12.8")
    if not (cuda/"bin/nvcc").is_file():
        raise RuntimeError("Existing CUDA 12.8 compiler unavailable; no environment migration performed")
    environment = dict(os.environ)
    environment.update(CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST="12.0", MAX_JOBS="2",
                       PATH=str(cuda/"bin")+":"+environment["PATH"])
    contract = {"source": str(args.source), "commit": commit, "license": "MIT; retained in source checkout",
        "compiler": subprocess.check_output([str(cuda/"bin/nvcc"), "--version"], text=True),
        "source_identity": [data_identity(args.source/name) for name in ("setup.py", "ssim.cu", "ext.cpp", "fused_ssim/__init__.py")],
        "install_performed": False, "existing_environment_modified": False, "architecture": "sm_120",
        "maximum_build_seconds": 600, "maximum_jobs": 2, "counted": False}
    atomic_json(args.output/"CONTRACT.json", contract)
    command = [sys.executable, "setup.py", "build_ext", "--build-temp", str(args.output/"temp"),
               "--build-lib", str(args.output/"lib")]
    started = time.monotonic()
    with (args.output/"build.log").open("w") as log:
        result = subprocess.run(command, cwd=args.source, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    receipt = {"status": "built_pending_numeric_validation" if result.returncode == 0 else "build_failed",
        "returncode": result.returncode, "command": command, "elapsed_seconds": time.monotonic()-started,
        "module_files": [data_identity(path) for path in (args.output/"lib").glob("*.so")],
        "existing_environment_modified": False, "training_acceleration_claim": False}
    atomic_json(args.output/"RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)
    if result.returncode:
        raise RuntimeError("Isolated fused SSIM build failed; original renderer remains available")


if __name__ == "__main__":
    main()
