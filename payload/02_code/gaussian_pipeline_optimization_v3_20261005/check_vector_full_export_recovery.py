"""Interrupt owned real exports and verify complete recovery in a new directory."""

import json
import multiprocessing.shared_memory as shared_memory
from pathlib import Path
import subprocess
import sys
import time

import psutil

from profile_vector_full_export import ROOT, BASE, read
from profile_sparse_ply_oracle import compare_outputs
from runtime import atomic_json, data_identity


def closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def stop_real_export(mode, output):
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).with_name("profile_vector_full_export.py")
    side = output / "partial"
    command = [sys.executable, "-B", str(code), "--mode", mode, "--output", str(side)]
    instances = {}
    with (output / "partial.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        root = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
        started = time.perf_counter()
        actual_count = 0
        while True:
            if process.poll() is not None:
                raise AssertionError("The real export exited before the registered interruption")
            parent = psutil.Process(process.pid)
            if parent.create_time() != root["creation_time"]:
                raise RuntimeError("The owned export process identity changed")
            for item in [parent, *parent.children(recursive=True)]:
                try:
                    row = {"pid": item.pid, "creation_time": item.create_time()}
                    instances[(row["pid"], row["creation_time"])] = row
                except psutil.NoSuchProcess:
                    continue
            folder = side / "export/dataset/images"
            actual_count = len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0
            if actual_count >= 17:
                parent.suspend()
                for item in parent.children(recursive=True):
                    try:
                        row = {"pid": item.pid, "creation_time": item.create_time()}
                        instances[(row["pid"], row["creation_time"])] = row
                    except psutil.NoSuchProcess:
                        continue
                break
            if time.perf_counter() - started > 120:
                process.terminate()
                process.wait(timeout=10)
                raise TimeoutError("The registered partial export did not reach17 within its budget")
            time.sleep(.02)
        header_path = side / "sampling_proof/SHARED_SEGMENTS_CREATED.json"
        headers = read(header_path)["headers"] if header_path.exists() else []
        terminated = []
        for row in reversed(list(instances.values())):
            if row == root:
                continue
            try:
                item = psutil.Process(row["pid"])
                if item.create_time() != row["creation_time"]:
                    continue
                item.terminate()
                terminated.append(item)
            except psutil.NoSuchProcess:
                continue
        _, alive = psutil.wait_procs(terminated, timeout=10)
        if alive:
            raise RuntimeError("An owned export descendant did not terminate")
        if parent.create_time() != root["creation_time"]:
            raise RuntimeError("The suspended owned parent identity changed")
        parent.terminate()
        returncode = process.wait(timeout=10)
        if returncode == 0:
            raise AssertionError("A forced incomplete export unexpectedly succeeded")
    if not all(closed(row) for row in instances.values()):
        raise AssertionError("An actual interrupted export instance still runs")
    for header in headers:
        try:
            segment = shared_memory.SharedMemory(name=header["name"], create=False)
        except FileNotFoundError:
            continue
        segment.close()
        raise AssertionError("A killed owned Windows shared segment remains available")
    if mode == "vector" and len(headers) != 5:
        raise AssertionError("The vector interruption did not exercise actual shared geometry")
    result = {"mode": mode, "target_file_threshold": 17, "actual_partial_JPEG_count_at_trigger": actual_count,
        "actual_nonzero_returncode": returncode, "instances": list(instances.values()), "headers": headers,
        "all_actual_instances_closed": True, "all_created_shared_names_unavailable": True,
        "not_exact17_images_or_normal_CPP_exception_boundary": True, "seconds": time.perf_counter() - started}
    atomic_json(output / "INTERRUPTION.json", result)
    return result


def main():
    started = time.perf_counter()
    output = BASE / "v0"
    output.mkdir(parents=True, exist_ok=False)
    saved = read(BASE / "c0/t0/native/CONTRACT.json")
    for key in ("sources", "inputs"):
        if saved[key] != [data_identity(row["path"]) for row in saved[key]]:
            raise ValueError("A first formal source or full input changed")
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    (snapshots / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    interruptions = [stop_real_export(mode, output / mode) for mode in ("native", "vector")]
    destination = output / "recovered"
    command = [sys.executable, "-B", str(Path(__file__).with_name("profile_vector_full_export.py")),
               "--mode", "vector", "--output", str(destination)]
    with (output / "recovered.log").open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        owner = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
        returncode = process.wait()
    if returncode or not closed(owner):
        raise RuntimeError("The complete new-directory recovery failed")
    result = read(destination / "RESULT.json")
    comparison = compare_outputs(BASE / "c0/t0/native/export", destination / "export")
    checks = {"both_real_partial_exports_nonzero_and_every_saved_instance_closed": all(r["all_actual_instances_closed"] for r in interruptions),
        "vector_actual_five_shared_maps_released_after_process_interruption": len(interruptions[1]["headers"]) == 5
            and interruptions[1]["all_created_shared_names_unavailable"],
        "full_new_directory_recovery_all_original_side_guards": all(result["checks"].values()),
        "full1050_recovery1046_bytes_four_scientific_JSON_exact": comparison["complete_byte_pairs"] == 1046 and comparison["complete_JSON_pairs"] == 4,
        "all_saved_full_original_sources_and_inputs_unchanged": all(saved[key] == [data_identity(row["path"]) for row in saved[key]] for key in ("sources", "inputs"))}
    if not all(checks.values()): raise AssertionError("A real full1050 recovery protection failed")
    atomic_json(output / "RESULT.json", {"status": "real_native_vector_partial_interrupt_and_new_directory_full1050_recovery_verified",
        "checks": checks, "interruptions": interruptions, "recovered_process": owner, "comparison": comparison,
        "source": data_identity(Path(__file__)), "complete_seconds": time.perf_counter() - started,
        "normal_CPP_filesystem_error_order_or_unseen_object_GT_not_proven": True, "new_optimization_count": 0})
    print(json.dumps({"status": "full1050_recovery_verified", "checks": checks, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
