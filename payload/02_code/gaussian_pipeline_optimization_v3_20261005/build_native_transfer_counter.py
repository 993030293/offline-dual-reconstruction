"""Build the isolated native meter using the existing CUDA ABI headers."""

from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    output = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001/build"
    output.mkdir(parents=True, exist_ok=False)
    source = CODE / "native_cuda_transfer_counter.c"
    header = Path("/usr/local/cuda-12.8/include/cuda_runtime_api.h")
    identities = [data_identity(path) for path in (source, header, Path(__file__))]
    shutil.copyfile(source, output / source.name)
    command = ["/usr/bin/gcc", "-std=c11", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror",
               "-I/usr/local/cuda-12.8/include", str(source), "-o", str(output / "native_transfer_counter.so"),
               "-ldl", "-lpthread"]
    child = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (output / "compiler.log").write_text(child.stdout + child.stderr, encoding="utf-8")
    atomic_json(output / "BUILD.json", {"command": command, "returncode": child.returncode,
        "sources": identities, "elapsed_seconds": time.perf_counter() - started,
        "artifact": data_identity(output / "native_transfer_counter.so") if child.returncode == 0 else None,
        "all_sources_unchanged": identities == [data_identity(path) for path in (source, header, Path(__file__))],
        "new_optimization_count": 0})
    if child.returncode:
        raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
