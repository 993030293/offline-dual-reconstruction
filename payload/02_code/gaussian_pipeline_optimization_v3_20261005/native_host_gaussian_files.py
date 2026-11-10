"""Keep the admitted PLY proof while dispatching owned NTFS file work to its host."""

import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import streamed_gaussian_ply_oracle_002 as original

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001"
HOST_PYTHON = ROOT / "05_envs/sfm_py311_v1/Scripts/python.exe"
HOST_WORKER = Path(__file__).with_name("native_host_file_worker.py")
ANCHOR = "            shutil.copyfile(path, oracle)\n            actual, expected = data_identity(path), data_identity(oracle)"


def windows_path(path):
    path = Path(path).resolve()
    path.relative_to(Path("/mnt/c"))
    return "C:\\" + str(path.relative_to(Path("/mnt/c"))).replace("/", "\\")


def admitted_path(source, target):
    if os.name != "posix" or not os.environ.get("WSL_INTEROP") or not HOST_PYTHON.is_file():
        return False
    try:
        source, target = Path(source), Path(target)
        source.resolve().relative_to(OUTPUT.resolve())
        target.resolve().relative_to(OUTPUT.resolve())
        if source.is_symlink() or target.is_symlink() or not source.is_file() or target.exists():
            return False
        if source.parent != target.parent or source.name != "point_cloud.ply" or target.name != "native_oracle.ply":
            return False
        if source.stat().st_size > original.MAXIMUM_BYTES or (source.parent / "HOST_IO_RESOURCE.json").exists():
            return False
        for parent in (source.parent, *source.parent.parents):
            if parent.is_symlink():
                return False
    except (OSError, ValueError):
        return False
    return True


def host_copy_and_identities(source, target, events):
    from runtime import data_identity
    source, target = Path(source), Path(target)
    if not admitted_path(source, target):
        events.append({"route": "original_before_execution", "source": str(source)})
        shutil.copyfile(source, target)
        return data_identity(source), data_identity(target)
    before = time.perf_counter()
    receipt = source.parent / "HOST_IO_RESOURCE.json"
    command = [str(HOST_PYTHON), "-B", windows_path(HOST_WORKER),
        "--source", windows_path(source), "--target", windows_path(target), "--receipt", windows_path(receipt)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=130)
    except BaseException:
        # The host worker has its own deadline; its real exit precedes propagation.
        process.communicate(timeout=140)
        raise
    if not receipt.is_file():
        raise RuntimeError("The actual host worker did not publish its resource identity: " + stderr.decode("utf-8", errors="replace"))
    row = json.loads(receipt.read_text(encoding="utf-8"))
    row["interop_process_returncode"] = process.returncode
    row["complete_interop_seconds"] = time.perf_counter() - before
    events.append(row)
    if process.returncode != 0 or row.get("status") != "completed":
        if row.get("errno") is not None:
            raise OSError(row["errno"], os.strerror(row["errno"]), str(target))
        raise RuntimeError("The actual host file operation failed: " + json.dumps(row, ensure_ascii=True))
    if json.loads(stdout.decode("utf-8")) != {key: value for key, value in row.items()
            if key not in ("interop_process_returncode", "complete_interop_seconds")}:
        raise RuntimeError("The worker result and durable receipt disagree")
    if not row["independent_file"]:
        raise AssertionError("The native PLY oracle requires a physically independent file")
    actual, expected = dict(row["source_identity"]), dict(row["target_identity"])
    actual["path"], expected["path"] = str(source), str(target)
    return actual, expected


def host_ply_model(base_class, events):
    source = inspect.getsource(original.streamed_ply_model)
    if source.count(ANCHOR) != 1:
        raise ValueError("The pinned admitted file-operation boundary changed")
    source = source.replace(ANCHOR, "            actual, expected = owned_file_work(path, oracle)", 1)
    namespace = dict(original.__dict__)
    namespace["owned_file_work"] = lambda path, oracle: host_copy_and_identities(path, oracle, events)
    exec(compile(source, "<owned_R127_native_host_file_boundary>", "exec"), namespace)
    return namespace["streamed_ply_model"](base_class)
