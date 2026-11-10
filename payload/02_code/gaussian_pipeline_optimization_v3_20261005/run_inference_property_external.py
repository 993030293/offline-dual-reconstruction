"""Run the independent public-input component with the unchanged owned launcher."""

import argparse
from pathlib import Path
import shutil
import sys
import time

import psutil

import run_inference_property_task as launch


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=("x0", "x1"), default="x0")
    args = parser.parse_args()
    output = launch.OUT / args.name
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    launch.atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()})
    paths = [Path(__file__), launch.CODE / "external_inference_properties.py", launch.CODE / "inference_property_workspace.py",
             Path(launch.__file__), launch.CODE / "run_bounded_journal_task.py", launch.CODE / "nvml_memory_monitor_002.py"]
    sources = [launch.data_identity(path) for path in paths]
    copies = output / "source_copies"
    copies.mkdir()
    for index, path in enumerate(paths):
        shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
    launch.run(output, "external_inference_properties.py", ["--output", launch.linux(output / "linux")])
    result = launch.read(output / "linux/RESULT.json")
    if not all(result["checks"].values()) or sources != [launch.data_identity(path) for path in paths]:
        raise AssertionError("The full public-input component or source protection failed")
    launch.atomic_json(output / "RESULT.json", {"status": "external360_component_passed", "checks": result["checks"],
        "sources": sources, "complete_seconds": time.perf_counter() - started, "new_optimization_count": 0})


if __name__ == "__main__":
    main()
