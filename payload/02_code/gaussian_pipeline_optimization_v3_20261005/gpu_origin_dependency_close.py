"""Verify only owned R118 diagnostic source, outputs, processes and cleanup."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gpu_origin_dependency_001"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def closed_process(pid, creation):
    try:
        current = psutil.Process(pid).create_time()
        return {"pid": pid, "recorded_creation_time": creation,
                "same_owned_process_alive": abs(current - creation) < 0.001,
                "PID_reused": abs(current - creation) >= 0.001}
    except psutil.NoSuchProcess:
        return {"pid": pid, "recorded_creation_time": creation,
                "same_owned_process_alive": False, "PID_reused": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() != OWN / "DIAGNOSTIC_CLOSURE.json" or args.output.exists():
        raise ValueError("Use the fresh owned diagnostic closure path")
    started = time.perf_counter()
    folder = OWN / "diagnostic_001"
    contract = load(folder / "CONTRACT.json")
    source_rows = []
    for row in contract["identities"]:
        current = identity(Path(row["path"]))
        if current != {key: row[key] for key in ("bytes", "sha256")}:
            raise AssertionError("An original frozen diagnostic source/input changed")
        source_rows.append({**row, "complete_identity_equal": True})
    snapshots = []
    for copied in (folder / "source_snapshot").iterdir():
        source = contract["identities"][int(copied.name[:2])]
        if identity(copied) != {key: source[key] for key in ("bytes", "sha256")}:
            raise AssertionError("A frozen diagnostic source snapshot changed")
        snapshots.append({"path": str(copied), "complete_source_identity_equal": True})
    close = load(folder / "RESOURCE_CLOSE.json")
    if not close["browserCloseAwaited"] or not close["serverCloseAwaited"]:
        raise AssertionError("Owned browser/server close did not complete")
    for row in close["cleanup"]:
        if row.get("contextBindings") != 0 or not row.get("workerObserversRestored") or row.get("GPUdrainFailure") is not None:
            raise AssertionError("Owned hooks/worker/GPU drain closure is incomplete")
        if not all(row.get(key) is True for key in ("numericGLObserverRestored", "generateObserversRestored", "diagnosticContextListenersRemoved")):
            raise AssertionError("Owned diagnostic observer closure is incomplete")
    lease = load(OWN / "LEASE_DIAGNOSTIC_001.json")
    if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
        raise AssertionError("Owned diagnostic lease did not drain and release")
    recorded = {f"{row['pid']}:{row['creation_time']}": closed_process(row["pid"], row["creation_time"])
                for row in close["processLedger"]}
    for prefix in ("guard", "child"):
        recorded[f"{lease[prefix + '_pid']}:{lease[prefix + '_creation_time']}"] = closed_process(
            lease[prefix + "_pid"], lease[prefix + "_creation_time"])
    if any(row["same_owned_process_alive"] for row in recorded.values()):
        raise AssertionError("A recorded owned R118 process is still alive")
    actual_files = [path for path in (folder / "actual").rglob("*") if path.is_file()]
    actual_rows = [{"path": str(path), "bytes": path.stat().st_size} for path in actual_files]
    for state_path in [path for path in actual_files if path.name == "STATE.json"]:
        state = load(state_path)
        expected = {"pixels.bin": state["width"] * state["height"] * 4, "camera.bin": 39 * 8,
                    "packed_gpu.bin": ((state["pointCount"] + 2047) // 2048) * 2048 * 16}
        for name, size in expected.items():
            if (state_path.parent / name).stat().st_size != size:
                raise AssertionError("An actual complete GPU/camera output size differs")
        if (state_path.parent / "ordering_gpu.bin").stat().st_size != (state_path.parent / "ordering_worker.bin").stat().st_size:
            raise AssertionError("Full GPU/worker ordering capacity differs")
    output_files = [path for path in OWN.rglob("*") if path.is_file()]
    output_bytes = sum(path.stat().st_size for path in output_files)
    if output_bytes > 30 * 1024**3 or psutil.disk_usage(OWN).free < 20 * 1024**3:
        raise AssertionError("Original R118 output/headroom resource gate failed")
    result = load(folder / "RESULT.json")
    payload = {"status": "diagnostic_source_resource_output_closure_passed", "identities": source_rows,
               "source_snapshot_count": len(snapshots), "source_snapshots": snapshots,
               "actual_files": actual_rows, "actual_file_count": len(actual_rows),
               "actual_bytes": sum(row["bytes"] for row in actual_rows),
               "complete_capture_states": sum(path.name == "STATE.json" for path in actual_files),
               "lease": lease, "all_recorded_owned_PIDs_closed": True, "recorded_owned_processes": list(recorded.values()),
               "process_coverage": "actual sampled owned browser tree and lease controller identities; no unsampled transient PID claim",
               "cleanup": close["cleanup"], "frozen_old_R115_unchanged": True,
               "diagnostic_result_status": result["status"], "no_new_algorithm_acceptance": True,
               "cost_pairs": 0, "output_files_before_closure": len(output_files), "output_bytes_before_closure": output_bytes,
               "closure_seconds": time.perf_counter() - started, "counted": False,
               "document_acceptance": "direct original text readback and consistency review only"}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "source_snapshot_count", "actual_file_count", "actual_bytes",
                                                  "all_recorded_owned_PIDs_closed", "counted")}))


if __name__ == "__main__":
    main()
