"""Compile private vector statistics with the existing Windows MSVC toolchain."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/vector_gray_stats_001")
    output.mkdir(parents=True, exist_ok=False)
    tool = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207")
    sdk = Path("C:/Program Files (x86)/Windows Kits/10")
    compiler = tool / "bin/Hostx64/x64/cl.exe"
    includes = [tool / "include", sdk / "Include/10.0.26100.0/ucrt"]
    libraries = [tool / "lib/x64", sdk / "Lib/10.0.26100.0/ucrt/x64", sdk / "Lib/10.0.26100.0/um/x64"]
    if not compiler.is_file() or not all(path.is_dir() for path in includes + libraries):
        raise FileNotFoundError("The installed vector-statistics compiler or SDK changed")
    source = Path(__file__).with_name("native_vector_gray_stats.cpp")
    paths = [Path(__file__), source, compiler]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in paths[:2]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    command = [str(compiler), "/nologo", "/LD", "/O2", "/MT", "/arch:AVX2", "/fp:precise", "/EHsc", "/std:c++17", "/W4",
               *["/I" + str(path) for path in includes], str(source), "/Fo" + str(output / "native_vector_gray_stats.obj"),
               "/Fe" + str(output / "native_vector_gray_stats.dll"), "/link", "/MACHINE:X64",
               *["/LIBPATH:" + str(path) for path in libraries]]
    environment = os.environ.copy()
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    atomic_json(output / "PROTOCOL.json", {"command": command, "sources": sources, "counted": False})
    try:
        with (output / "compiler.log").open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=output, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            raise RuntimeError("The private vector compilation failed with code " + str(result.returncode))
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("An actual vector compiler source changed")
        atomic_json(output / "RESULT.json", {"status": "built_numerical_verification_pending", "sources": sources,
            "DLL": data_identity(output / "native_vector_gray_stats.dll"), "seconds": time.perf_counter() - started, "counted": False})
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"error": repr(error), "seconds": time.perf_counter() - started, "counted": False})
        raise


if __name__ == "__main__":
    main()
