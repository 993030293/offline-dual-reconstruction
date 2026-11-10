"""Compile the private cache writer and close only its own MSVC telemetry child."""

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
    output = parser.parse_args().output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_cache_writer_001")
    output.mkdir(parents=True, exist_ok=False)
    tool = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207")
    sdk = Path("C:/Program Files (x86)/Windows Kits/10")
    compiler = tool / "bin/Hostx64/x64/cl.exe"
    includes = [tool / "include", *[sdk / ("Include/10.0.26100.0/" + name) for name in ("ucrt", "um", "shared")]]
    libraries = [tool / "lib/x64", sdk / "Lib/10.0.26100.0/ucrt/x64", sdk / "Lib/10.0.26100.0/um/x64"]
    if not compiler.is_file() or not all(path.is_dir() for path in includes + libraries):
        raise FileNotFoundError("The existing private cache compiler or SDK changed")
    source = Path(__file__).with_name("native_quality_cache_writer.cpp")
    paths = [Path(__file__), source, compiler]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in paths[:2]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    command = [str(compiler), "/nologo", "/LD", "/O2", "/MT", "/EHsc", "/std:c++17", "/W4",
        *["/I" + str(path) for path in includes], str(source), "/Fo" + str(output / "native_quality_cache_writer.obj"),
        "/Fe" + str(output / "native_quality_cache_writer.dll"), "/link", "/MACHINE:X64", "kernel32.lib",
        *["/LIBPATH:" + str(path) for path in libraries]]
    environment = os.environ.copy()
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    atomic_json(output / "PROTOCOL.json", {"command": command, "sources": sources, "counted": False})
    try:
        with (output / "compiler.log").open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=output, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            compiler_born = psutil.Process(process.pid).create_time()
            while process.poll() is None:
                if time.perf_counter() - started > 120:
                    process.terminate()
                    process.wait(timeout=10)
                    raise TimeoutError("The private compiler exceeded its existing budget")
                time.sleep(.05)
        if process.returncode:
            raise RuntimeError("The private cache compilation failed with code " + str(process.returncode))
        end = time.time()
        telemetry = []
        for item in psutil.process_iter(["pid", "name", "ppid", "create_time"]):
            if item.info["name"].lower() != "vctip.exe" or item.info["ppid"] != process.pid or not compiler_born <= item.info["create_time"] <= end:
                continue
            expected = compiler.parent / "vctip.exe"
            executable = Path(item.exe())
            if executable != expected or item.create_time() != item.info["create_time"]:
                raise ValueError("A candidate compiler telemetry identity changed")
            saved = {"pid": item.pid, "creation_time": item.create_time(), "exe": str(executable),
                     "compiler_pid": process.pid, "compiler_creation_time": compiler_born}
            item.terminate()
            item.wait(timeout=10)
            saved["exact_instance_exited"] = not item.is_running()
            if not saved["exact_instance_exited"]:
                raise ValueError("The exact owned compiler telemetry instance did not exit")
            telemetry.append(saved)
        atomic_json(output / "BUILD_TELEMETRY_CLOSE.json", {"owned_instances": telemetry, "source": data_identity(Path(__file__)),
            "strict_parent_birth_window_exe_and_PID_only": True, "new_optimization_count": 0})
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("An actual private cache build input changed")
        atomic_json(output / "RESULT.json", {"status": "built_numerical_and_interface_verification_pending", "sources": sources,
            "DLL": data_identity(output / "native_quality_cache_writer.dll"), "seconds": time.perf_counter() - started, "counted": False})
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"error": repr(error), "seconds": time.perf_counter() - started, "counted": False})
        raise


if __name__ == "__main__":
    main()
