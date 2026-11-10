"""Build pinned gsplat into an isolated library with no environment installation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from runtime import atomic_json, data_identity


PIN = "937e29912570c372bed6747a5c9bf85fed877bae"
GLM_PIN = "33b4a621a697a305bc3a7610d290677b96beb181"
WHEELS = {"rich": "13.9.4", "pygments": "2.19.1", "markdown_it_py": "3.0.0", "mdurl": "0.1.2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source, args.output, args.wheels = args.source.resolve(), args.output.resolve(), args.wheels.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for path, expected in [(args.source, PIN), (args.source / "gsplat/cuda/csrc/third_party/glm", GLM_PIN)]:
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-c", "core.autocrlf=true", "-C", str(path), "status",
            "--porcelain", "--untracked-files=no", "--ignore-submodules=dirty"], text=True).strip()
        if actual != expected or dirty:
            raise ValueError("Pinned source or submodule identity changed")
    dependencies = args.output / "dependencies"
    dependencies.mkdir()
    identities = []
    for name, version in WHEELS.items():
        candidates = list(args.wheels.glob(name+"-"+version+"-*-none-any.whl"))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one explicitly versioned pure-Python wheel")
        wheel = candidates[0]
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                target = (dependencies / member.filename).resolve()
                if dependencies != target and dependencies not in target.parents:
                    raise ValueError("Wheel path escapes the isolated dependency directory")
            archive.extractall(dependencies)
        identities.append(data_identity(wheel))
    cuda = Path("/usr/local/cuda-12.8")
    environment = dict(os.environ, CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST="12.0", MAX_JOBS="2")
    environment["PATH"] = str(Path(sys.executable).parent)+":"+str(cuda / "bin")+":"+environment["PATH"]
    contract = {"source": str(args.source), "commit": PIN, "glm_commit": GLM_PIN, "version": "1.5.3",
        "license": "Apache-2.0; original license and source retained", "cuda": str(cuda), "arch": "sm_120",
        "max_build_workers": 2, "maximum_seconds": 900, "dependency_wheels": identities,
        "environment_installation": False, "source_modified": False, "input_normalization_or_dataset_split_change": False,
        "source_identity": data_identity(Path(__file__)), "counted": False}
    atomic_json(args.output / "CONTRACT.json", contract)
    command = [sys.executable, "setup.py", "build_ext", "--build-temp", str(args.output / "temp"),
               "--build-lib", str(args.output / "lib")]
    started = time.monotonic()
    with (args.output / "build.log").open("w") as log:
        process = subprocess.Popen(command, cwd=args.source, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        atomic_json(args.output / "STATE.json", {"status": "building", "pid": process.pid, "controller_pid": os.getpid()})
        try:
            code = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            atomic_json(args.output / "STATE.json", {"status": "build_timeout", "elapsed_seconds": time.monotonic()-started})
            raise
    package = args.output / "lib/gsplat"
    modules = list(package.glob("csrc*.so"))
    if code == 0 and len(modules) == 1:
        shutil.copytree(args.source / "gsplat", package, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"))
        shutil.copyfile(args.source / "LICENSE", args.output / "LICENSE")
    else:
        atomic_json(args.output / "STATE.json", {"status": "build_failed", "returncode": code,
            "elapsed_seconds": time.monotonic()-started})
        raise RuntimeError("Isolated gsplat build did not produce its expected module")
    result = {"status": "built_pending_projection_and_training", "returncode": code,
        "elapsed_seconds": time.monotonic()-started, "command": command,
        "modules": [data_identity(path) for path in modules], "library": str(args.output / "lib"),
        "dependencies": str(dependencies), "counted": False}
    atomic_json(args.output / "RESULT.json", result)
    atomic_json(args.output / "STATE.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
