"""Close only the frozen R115 negative evidence and owned historical processes."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import psutil


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/origin_dependency_001"
NODES = ("generic_001", "generic_002", "generic_003", "lifecycle_001", "lifecycle_002")
RECEIPTS = ("LEASE_GENERIC_001.json", "LEASE_GENERIC_002.json", "LEASE_GENERIC_003.json",
            "LEASE_CONTEXT_001.json", "LEASE_CONTEXT_002.json")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def data_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def process_identity(pid, creation):
    try:
        process = psutil.Process(pid)
        current = process.create_time()
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
    if args.output.resolve() != OUTPUT / "CLOSURE.json" or args.output.exists():
        raise ValueError("Closure must use the fresh dedicated R115 receipt")
    started = time.perf_counter()
    expected = {}
    snapshots = []
    nodes = {}
    for node in NODES:
        folder = OUTPUT / node
        contract = load(folder / "CONTRACT.json")
        for row in contract["sourcesAndInputs"]:
            previous = expected.setdefault(row["path"], row)
            if previous != row:
                raise AssertionError("An unchanged frozen source/input received different identities")
        for snapshot in (folder / "source_snapshot").iterdir():
            index = int(snapshot.name[:2])
            original = contract["sourcesAndInputs"][index]
            copied = data_identity(snapshot)
            if copied["bytes"] != original["bytes"] or copied["sha256"] != original["sha256"]:
                raise AssertionError("A frozen source snapshot changed")
            snapshots.append({"path": str(snapshot), "complete_source_bytes_equal": True})
        result = load(folder / "RESULT.json")
        closing = load(folder / "GENERIC_RESOURCE_CLOSE.json")
        actual = folder / "actual"
        actual_files = list(actual.rglob("*")) if actual.exists() else []
        actual_files = [path for path in actual_files if path.is_file()]
        states = [path for path in actual_files if path.name == "STATE.json"]
        bins = [path for path in actual_files if path.suffix == ".bin"]
        nodes[node] = {"original_status": result["status"],
                       "browser_closed_receipt": closing["browserClosed"],
                       "server_closed_receipt": closing["serverClosed"],
                       "original_cleanup_rows": closing["restored"],
                       "actual_files": len(actual_files), "complete_capture_states": len(states),
                       "binary_publications": len(bins),
                       "actual_bytes": sum(path.stat().st_size for path in actual_files)}
        if not closing["browserClosed"] or not closing["serverClosed"]:
            raise AssertionError("An owned browser/server did not close")
    current = []
    for name, row in expected.items():
        actual = data_identity(Path(name))
        actual["path"] = name
        if actual != row:
            raise AssertionError(f"Frozen source/input changed: {name}")
        current.append(actual)
    leases = []
    owned_processes = []
    for name in RECEIPTS:
        row = load(OUTPUT / name)
        if row["status"] != "child_exited" or not row.get("lease_released_utc"):
            raise AssertionError("A measured historical node did not drain and release")
        for prefix in ("guard", "child"):
            owned_processes.append(process_identity(row[prefix + "_pid"], row[prefix + "_creation_time"]))
        leases.append({"path": str(OUTPUT / name), "owner": row["owner"],
                       "node_elapsed_seconds": row["child_elapsed_seconds"],
                       "returncode": row["returncode"], "released_utc": row["lease_released_utc"],
                       "wait_seconds": row["wait_seconds"]})
    if any(row["same_owned_process_alive"] for row in owned_processes):
        raise AssertionError("A recorded R115 controller/child remains alive")
    files = [path for path in OUTPUT.rglob("*") if path.is_file()]
    byte_count = sum(path.stat().st_size for path in files)
    if byte_count > 30 * 1024 ** 3 or psutil.disk_usage(str(OUTPUT)).free < 20 * 1024 ** 3:
        raise AssertionError("The original output/headroom gate failed")
    primary = load(OUTPUT / "generic_003/GENERIC_PRIMARY_RESULT.json")
    if len(primary["completed"]) != 118 or primary["fullSemanticAcceptance"]:
        raise AssertionError("Primary evidence scope changed")
    negative = load(OUTPUT / "lifecycle_002/GENERIC_FAILURE.json")
    if negative["current"]["name"] != "scene_real_context_restore_rotate":
        raise AssertionError("The first required negative lifecycle node changed")
    original_capture = load(OUTPUT / "lifecycle_002/actual/native_scene_real_context_restore_rotate/STATE.json")
    if original_capture["errors"] or not original_capture["sourceSortingClosed"]:
        raise AssertionError("The original route did not pass the same recovery observation")
    final_cleanup = nodes["lifecycle_002"]["original_cleanup_rows"]
    if any(row.get("contextBindings") != 0 or not row.get("workerObserversRestored") or
           row.get("GPUdrainFailure") is not None for row in final_cleanup):
        raise AssertionError("The final owned contexts/workers/GPU drain did not close")
    payload = {"status": "negative_source_resource_output_closure_passed",
               "frozen_source_input_count": len(current), "identities": current,
               "complete_source_snapshot_count": len(snapshots), "source_snapshots": snapshots,
               "nodes": nodes, "leases": leases, "owned_processes": owned_processes,
               "all_recorded_owned_PIDs_closed": True, "final_context_bindings_zero": True,
               "final_worker_observers_restored": True, "final_actual_GPU_drains_passed": True,
               "owned_output_files_before_closure": len(files), "owned_output_bytes_before_closure": byte_count,
               "actual_GPU_check_node_seconds": sum(row["node_elapsed_seconds"] for row in leases),
               "closure_seconds": time.perf_counter() - started,
               "performance_pairs": 0, "counted": False,
               "old_setup_cleanup_failure_preserved": "lifecycle_001 native GPU drain failed; its private page/browser was destroyed and both controller PIDs are closed",
               "browser_PID_ledger_not_captured": True,
               "source_input_hashes_prove_identity_only": True,
               "document_acceptance": "direct text readback and consistency review only"}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "frozen_source_input_count",
                     "complete_source_snapshot_count", "owned_output_bytes_before_closure",
                     "all_recorded_owned_PIDs_closed", "counted")}))


if __name__ == "__main__":
    main()
