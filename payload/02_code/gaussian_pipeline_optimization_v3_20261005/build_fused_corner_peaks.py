"""Compile the private finite-response peak extractor with the installed SDK."""

import os
from pathlib import Path
import subprocess
import time


def build_library(output):
    from runtime import atomic_json, data_identity
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tool = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207")
    sdk = Path("C:/Program Files (x86)/Windows Kits/10")
    compiler = tool / "bin/Hostx64/x64/cl.exe"
    includes = [tool / "include", sdk / "Include/10.0.26100.0/ucrt"]
    libraries = [tool / "lib/x64", sdk / "Lib/10.0.26100.0/ucrt/x64", sdk / "Lib/10.0.26100.0/um/x64"]
    if not compiler.is_file() or not all(path.is_dir() for path in includes + libraries):
        raise FileNotFoundError("The current compiler or SDK changed")
    source = Path(__file__).with_name("native_fused_corner_peaks.cpp")
    paths = [Path(__file__), source, compiler]
    identities = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in paths[:2]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    library = output / "native_fused_corner_peaks.dll"
    command = [str(compiler), "/nologo", "/LD", "/O2", "/MT", "/arch:AVX2", "/fp:precise", "/EHsc", "/std:c++17", "/W4",
        *["/I" + str(path) for path in includes], str(source), "/Fo" + str(output / "native_fused_corner_peaks.obj"),
        "/Fe" + str(library), "/link", "/MACHINE:X64", *["/LIBPATH:" + str(path) for path in libraries]]
    environment = os.environ.copy()
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    atomic_json(output / "PROTOCOL.json", {"command": command, "sources": identities, "new_optimization_count": 0})
    began = time.perf_counter()
    with (output / "compiler.log").open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=output, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
    atomic_json(output / "RESULT.json", {"status": "built_generic_pending" if result.returncode == 0 else "failed_compile",
        "returncode": result.returncode, "sources": identities, "actual_compile_seconds": time.perf_counter() - began, "new_optimization_count": 0})
    if result.returncode or identities != [data_identity(path) for path in paths]:
        raise RuntimeError("The private peak-extractor build or source identity failed")
    return library
