"""Build one source-pinned private Windows sampling DLL with the installed toolchain."""

import argparse
import json
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    own = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001"
    args.output = args.output.resolve()
    args.output.relative_to(own.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()})
    tool = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207")
    sdk = Path("C:/Program Files (x86)/Windows Kits/10")
    compiler = tool / "bin/Hostx64/x64/cl.exe"
    include = [tool / "include", sdk / "Include/10.0.26100.0/ucrt"]
    libraries = [tool / "lib/x64", sdk / "Lib/10.0.26100.0/ucrt/x64", sdk / "Lib/10.0.26100.0/um/x64"]
    if not compiler.is_file() or not all(path.is_dir() for path in include + libraries):
        raise FileNotFoundError("The already verified installed MSVC/SDK inputs changed")
    source = Path(__file__).with_name("native_camera_sampling.cpp")
    identities = [data_identity(path) for path in (Path(__file__), source, compiler)]
    command = [str(compiler), "/nologo", "/LD", "/O2", "/MT", "/fp:strict", "/EHsc", "/std:c++17", "/W4",
               *["/I"+str(path) for path in include], str(source), "/Fo"+str(args.output / "native_camera_sampling.obj"),
               "/Fe"+str(args.output / "native_camera_sampling.dll"), "/link", "/MACHINE:X64",
               *["/LIBPATH:"+str(path) for path in libraries]]
    environment = os.environ.copy()
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    atomic_json(args.output / "PROTOCOL.json", {"sources": identities, "command": command,
        "strict_float_source_order_no_contract": True, "existing_toolchain_no_install": True, "counted": False})
    with (args.output / "compiler.log").open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=args.output, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        atomic_json(args.output / "FAILURE.json", {"returncode": result.returncode, "seconds": time.perf_counter()-started, "counted": False})
        raise RuntimeError("The private native sampling compilation failed")
    if identities != [data_identity(path) for path in (Path(__file__), source, compiler)]:
        raise RuntimeError("The private compiler source changed during build")
    receipt = {"status": "private_native_DLL_built_numerical_verification_pending", "sources": identities,
        "DLL": data_identity(args.output / "native_camera_sampling.dll"), "seconds": time.perf_counter()-started,
        "command": command, "counted": False}
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
