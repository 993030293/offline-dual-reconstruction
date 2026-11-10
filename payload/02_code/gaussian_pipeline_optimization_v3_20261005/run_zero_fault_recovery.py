"""Keep actual complete failures and recover only through a new full entry."""

from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

import run_native_zero_feature_campaign as parent
from runtime import atomic_json, data_identity

OUT = parent.OUT / "v3"


def run(mode, entry, label):
    leaf, node = OUT / label, OUT / (label + "_node")
    command = parent.command(leaf, node, mode)
    original_node = str(parent.CODE / "native_zero_feature_node.py")
    original_entry = "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_native_feature_zero.py"
    if command.count(original_node) != 1 or command.count(original_entry) != 1:
        raise ValueError("The protected full recovery command boundary changed")
    command[command.index(original_node)] = str(parent.CODE / "native_zero_checked_node.py")
    command[command.index(original_entry)] = original_entry.replace("profile_native_feature_zero.py", entry)
    begin = time.perf_counter()
    with (OUT / (label + ".log")).open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        owner = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
        code = process.wait()
    row = {"owner": owner, "actual_spawn_to_exit_seconds": time.perf_counter() - begin,
           "returncode": code, "leaf": str(leaf), "node": str(node)}
    atomic_json(OUT / (label + "_EXIT.json"), row)
    if code:
        raise RuntimeError("The original protected fault/recovery diagnostic failed")
    controller = parent.read(node / "NODE.json")
    if controller["status"] != "completed" or controller["live_Windows_after_wait"] or any(value["alive"] for value in controller["actual_Linux_after_wait"]):
        raise AssertionError("An owned full fault/recovery process remained live")
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[parent.CODE / name for name in ("profile_native_zero_fault.py",
        "native_zero_checked_node.py", "profile_native_zero_bounded.py", "profile_native_feature_zero.py",
        "native_feature_zero.py", "native_feature_zero_v2.py", "check_zero_transfer_contracts.py")]]
    ids = [data_identity(path) for path in sources]
    copies = OUT / "source_snapshot"
    copies.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))
    sides = {mode: run(mode, "profile_native_zero_fault.py", "fault_" + mode) for mode in ("native", "direct")}
    failed = {mode: parent.read(Path(row["leaf"]) / "FAULT_RESULT.json") for mode, row in sides.items()}
    checks = {"both_complete_fault_original_checks": all(all(row["checks"].values()) for row in failed.values()),
              "original_actual_error_and_model_prefix_exact": failed["native"]["error"] == failed["direct"]["error"]
                 and failed["native"]["evidence"]["actual_original_model_prefix"] == failed["direct"]["evidence"]["actual_original_model_prefix"],
              "all_original_full_inputs_exact": failed["native"]["inputs"] == failed["direct"]["inputs"]}
    if not all(checks.values()):
        raise AssertionError("The complete original native fault prefix changed")
    private_copy = OUT / "private_recovery_copy"
    private_copy.mkdir()
    removed = []
    for name in ("input.ply", "cameras.json"):
        source = Path(sides["direct"]["leaf"]) / "scene" / name
        destination = private_copy / name
        shutil.copyfile(source, destination)
        if destination.resolve().parent != private_copy.resolve() or destination.is_symlink():
            raise ValueError("A recoverable copy escaped its exact private directory")
        identity = data_identity(destination)
        destination.unlink()
        removed.append({"generated_private_copy": identity, "removed": not destination.exists(),
                        "original_failed_artifact_retained": source.is_file()})
    recovered = run("direct", "profile_native_zero_bounded.py", "recovered")
    comparison = parent.compare(parent.OUT / "c2/t0/native", Path(recovered["leaf"]))
    checks.update(actual_two_private_generated_files_removed_and_originals_retained=len(removed) == 2 and all(row["removed"] and row["original_failed_artifact_retained"] for row in removed),
                  actual_fresh_complete1016_recovery_all_science_exact=all(comparison["checks"].values()),
                  physical_sources_unchanged=ids == [data_identity(path) for path in sources])
    work = sum(row["extra_actual_native_HtoD_bytes"] for row in failed.values()) + parent.read(Path(recovered["leaf"]) / "ZERO_TRANSFER_TRACE.json")["actual_native_HtoD_bytes"]
    atomic_json(OUT / "RESULT.json", {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "sources": ids, "actual_fault_sides": sides, "private_removed": removed, "recovery": recovered,
        "comparison": comparison, "extra_actual_native_HtoD_bytes": work, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("An actual native zero-feature recovery check failed")


if __name__ == "__main__":
    main()
