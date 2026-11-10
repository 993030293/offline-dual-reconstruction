"""Build the private integer gray moments DLL with the installed compiler."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gm001")
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()})
    tool = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207")
    sdk = Path("C:/Program Files (x86)/Windows Kits/10")
    compiler = tool / "bin/Hostx64/x64/cl.exe"
    includes = [tool / "include", sdk / "Include/10.0.26100.0/ucrt"]
    libraries = [tool / "lib/x64", sdk / "Lib/10.0.26100.0/ucrt/x64", sdk / "Lib/10.0.26100.0/um/x64"]
    if not compiler.is_file() or not all(path.is_dir() for path in includes + libraries):
        raise FileNotFoundError("The existing compiler or SDK input changed")
    source = Path(__file__).with_name("native_gray_moments.cpp")
    paths = [Path(__file__), source, compiler]
    identities = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in paths[:2]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    command = [str(compiler), "/nologo", "/LD", "/O2", "/MT", "/fp:strict", "/EHsc", "/std:c++17", "/W4",
               *["/I" + str(path) for path in includes], str(source), "/Fo" + str(output / "native_gray_moments.obj"),
               "/Fe" + str(output / "native_gray_moments.dll"), "/link", "/MACHINE:X64",
               *["/LIBPATH:" + str(path) for path in libraries]]
    environment = os.environ.copy()
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    atomic_json(output / "PROTOCOL.json", {"sources": identities, "command": command, "existing_toolchain_no_install": True,
        "integer_only_scan_and_frozen_python_original_float_reconstruction": True, "counted": False})
    try:
        with (output / "compiler.log").open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=output, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            raise RuntimeError("The private gray moments compilation failed with code " + str(result.returncode))
        if identities != [data_identity(path) for path in paths]:
            raise RuntimeError("An owned compiler source changed during build")
        receipt = {"status": "built_numerical_verification_pending", "sources": identities,
                   "DLL": data_identity(output / "native_gray_moments.dll"), "seconds": time.perf_counter() - started, "counted": False}
        atomic_json(output / "RESULT.json", receipt)
        print({"status": receipt["status"], "seconds": receipt["seconds"], "counted": False}, flush=True)
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"error_type": type(error).__name__, "error": str(error),
                   "seconds": time.perf_counter() - started, "counted": False})
        raise


if __name__ == "__main__":
    main()
