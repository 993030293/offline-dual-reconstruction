"""Close the R118 complete-consumer screen without claiming formal acceptance."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OWN = OUT / "gpu_origin_dependency_001"
PROFILE = OWN / "root_profile_001"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def process_state(pid, creation):
    try:
        current = psutil.Process(pid).create_time()
        alive = abs(current - creation) < 0.001
        return {"pid": pid, "creation_time": creation, "same_owned_instance_alive": alive,
                "PID_reused": not alive}
    except psutil.NoSuchProcess:
        return {"pid": pid, "creation_time": creation, "same_owned_instance_alive": False,
                "PID_reused": False}


def owned_child(rows, owner):
    return any(row["pid"] == owner["child_pid"]
               and abs(row["creation_time"] - owner["child_creation_time"]) < 0.001 for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease-receipt", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination != OWN / "COMPLETE_SCREEN_CLOSURE_001.json" or destination.exists():
        raise ValueError("Use the fresh dedicated R118 screen closure destination")
    lease_receipt = args.lease_receipt.resolve()
    if lease_receipt != OUT / "gpu_origin_dependency_review_003/LEASE.json":
        raise ValueError("Use the separate owned review lease")
    started = time.perf_counter()
    current = psutil.Process()
    ancestry = [{"pid": process.pid, "creation_time": process.create_time()}
                for process in [current] + current.parents()]
    execution = lease_receipt.parent / "EXECUTION.json"
    if execution.exists():
        raise FileExistsError("Preserve each review execution receipt")
    (lease_receipt.parent / "REVIEWER_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    execution.write_text(json.dumps({"pid": current.pid, "creation_time": current.create_time(),
                         "actual_self_and_parent_identities": ancestry}, indent=2), encoding="utf-8")
    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    result = load(PROFILE / "PROFILE_RESULT.json")
    contract = load(PROFILE / "CONTRACT.json")
    plan = load(PROFILE / "PROFILE_PLAN.json")
    close = load(PROFILE / "PROFILE_RESOURCE_CLOSE.json")
    require("complete_profile_is_diagnostic_with_no_formal_or_acceptance_claim",
            result["status"] == "complete_fresh_pair_profile_bytes_and_cleanup_passed_not_formal_acceptance"
            and result["formalPairsExecuted"] == 0 and not result["counted"]
            and contract["formalPairsExecuted"] == 0 and not contract["counted"])
    require("V2_speed_and_all_original_cost_contracts_unchanged",
            contract["minimumPhaseReduction"] == 0.05 and contract["freshProcessPairs"] == 3
            and contract["maximumEachWholeRatio"] == 1.1 and contract["wholeMedianNonregression"]
            and contract["supplementalGPUGuardsRequired"] and contract["fullSourceInputHashesOnEachLeaf"])
    require("all_original118_steps_on_each_fresh_side_and_comparison",
            result["originalCompleteSteps"] == len(plan["steps"]) == len(result["completed"]) == 118
            and plan["freshBrowserOnEachSide"] and plan["modesInOrder"] == ["native", "candidate"]
            and plan["originalStepsCompleteAndUnchanged"])
    current_identities = []
    for row in contract["sourcesAndInputs"]:
        expected = {name: row[name] for name in ("bytes", "sha256")}
        require("complete_source_input_identity_" + str(len(current_identities)), identity(Path(row["path"])) == expected)
        current_identities.append(row)
    snapshots = []
    for snapshot in sorted((PROFILE / "source_snapshot").iterdir()):
        row = contract["sourcesAndInputs"][int(snapshot.name.split("_", 1)[0])]
        require("executed_source_snapshot_" + str(len(snapshots)),
                identity(snapshot) == {name: row[name] for name in ("bytes", "sha256")})
        snapshots.append(str(snapshot))
    byte_comparisons = 0
    actual_science_files = []
    for mode in ("native", "candidate"):
        rows = result["sides"][mode]["rows"]
        require(mode + "_all_registered_steps", [row["step"] for row in rows] == plan["steps"])
        gpu = result["sides"][mode]["hardware"]
        require(mode + "_actual_hardware_WebGL2", gpu["WebGL2"] and "NVIDIA" in gpu["renderer"] and "SwiftShader" not in gpu["renderer"])
        require(mode + "_original_resource_budget", all(row["resources"]["rss"] <= 8 * 1024**3
                and row["resources"]["usedMiB"] <= 12288 and row["resources"]["freeMiB"] >= 2048 for row in rows))
        for row in rows:
            state = row["row"]
            folder = PROFILE / "actual" / state["label"]
            published = load(folder / "STATE.json")
            require(mode + "_state_" + state["label"], published == state and state["maxSh"] == state["numSh"] == 3
                    and state["sourceSortingClosed"] and not state["errors"] and state["nativeIntegrityChecks"])
            for name, size in {
                "pixels.bin": state["width"] * state["height"] * 4,
                "camera.bin": 39 * 8,
                "packed_gpu.bin": ((state["pointCount"] + 2047) // 2048) * 2048 * 16,
            }.items():
                require(mode + "_size_" + state["label"] + "_" + name, (folder / name).stat().st_size == size)
            require(mode + "_complete_ordering_capacity_" + state["label"],
                    (folder / "ordering_gpu.bin").stat().st_size == (folder / "ordering_worker.bin").stat().st_size)
    for index, row in enumerate(result["completed"]):
        require("full_comparison_step_" + str(index), row["step"] == plan["steps"][index]
                and set(row["check"]) >= {"pixels.bin", "ordering_gpu.bin", "ordering_worker.bin", "packed_gpu.bin", "camera.bin"}
                and all(value["completeBytesEqual"] for value in row["check"].values()))
        byte_comparisons += len(row["check"])
    actual_science_files = [path for path in (PROFILE / "actual").rglob("*") if path.is_file()]
    require("all_actual_capture_files_retained", len(actual_science_files) == 1432)
    require("both_hooks_worker_GPU_browser_and_context_closed", len(close["cleanup"]) == 2
            and close["serverCloseAwaited"] and all(row["browserCloseAwaited"] and row["contextCloseAwaited"]
            and row["restoreError"] is None and row["restored"]["contextBindings"] == 0
            and row["restored"]["workerObserversRestored"] and row["restored"]["GPUdrainFailure"] is None
            and row["restored"]["nativeMeshPrototypePreserved"] for row in close["cleanup"]))
    public_fee = sum(result["commonMeasuredSeconds"].values()) / 2
    recomputed = {mode: result["sides"][mode]["wholeLeafSeconds"] + public_fee for mode in ("native", "candidate")}
    reduction = 1 - recomputed["candidate"] / recomputed["native"]
    require("complete_screen_measured_times_exactly_recomputed", public_fee == result["commonMeasuredFeeEach"]
            and recomputed == result["rawCompleteConsumerSeconds"] and reduction == result["rawCompleteConsumerReduction"])
    require("negative_screen_original5percent_gate_failed", reduction < 0.05)
    lease_paths = sorted(OWN.glob("*LEASE*.json"))
    fee_rows = []
    process_rows = {(row["pid"], row["creation_time"]): row for row in close["processLedger"]}
    for path in lease_paths:
        lease = load(path)
        require("owned_lease_released_" + path.stem, lease["status"] == "child_exited" and bool(lease.get("lease_released_utc")))
        fee_rows.append({"path": str(path), "complete_child_seconds": lease["child_elapsed_seconds"],
                         "returncode": lease["returncode"], "lock_wait_seconds_excluded": lease["wait_seconds"]})
        for prefix in ("guard", "child"):
            pid, creation = lease[prefix + "_pid"], lease[prefix + "_creation_time"]
            process_rows[(pid, creation)] = {"pid": pid, "creation_time": creation}
    processes = [process_state(*key) for key in process_rows]
    require("all_sampled_and_recorded_R118_owned_instances_exited", not any(row["same_owned_instance_alive"] for row in processes))
    require("unchanged_historical_four_step_and124_primary_evidence_retained",
            load(OWN / "root_smoke_002/RESULT.json")["counted"] is False
            and load(OWN / "root_primary_001/RESULT.json")["counted"] is False
            and len(load(OWN / "root_primary_001/GENERIC_PRIMARY_RESULT.json")["completed"]) == 124)
    owner = load(OUT / "coordination_001/LOCK_OWNER.json")
    require("current_review_owns_shared_lease_after_profile_released", owner["status"] == "running"
            and owner["owner"] == "root_R118_complete_screen_review_003" and owned_child(ancestry, owner))
    require("process_identity_reuse_wrong_PID_and_empty_ancestry_rejected", not owned_child([], owner)
            and not owned_child(ancestry, {**owner, "child_creation_time": owner["child_creation_time"] + 1})
            and not owned_child(ancestry, {**owner, "child_pid": -1}))
    files = [path for path in OWN.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    require("original_owned_output_and_disk_headroom_preserved", total <= 30 * 1024**3 and psutil.disk_usage(str(OWN)).free >= 20 * 1024**3)
    source_snapshot = OWN / "COMPLETE_SCREEN_REVIEWER_SOURCE_001.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    payload = {
        "status": "closed_negative_complete_consumer_screen_below5percent_no_formal_acceptance",
        "candidate": "R118_same_unaccepted_R115_dependency_family", "review_kind": "root_self_review_no_subagents",
        "accepted_count_contribution": 0, "counted": False, "checks": checks,
        "current_source_input_identities": current_identities, "executed_source_snapshots": snapshots,
        "complete_steps_each": 118, "actual_full_byte_comparisons": byte_comparisons,
        "actual_science_files": len(actual_science_files), "actual_science_bytes": sum(path.stat().st_size for path in actual_science_files),
        "screen_raw_complete_consumer_seconds": recomputed, "screen_raw_complete_reduction": reduction,
        "screen_not_formal_pairs_or_median": True, "known_R118_actual_child_fee_rows": fee_rows,
        "known_R118_actual_child_fees_including_screen_seconds": sum(row["complete_child_seconds"] for row in fee_rows),
        "old_R115_costs_unchanged_and_not_waived": True, "no_complete_formal_fee_acceptance_claim": True,
        "recorded_owned_processes": processes, "recorded_owned_instances_closed": len(processes),
        "process_coverage": "actual sampled browser trees and recorded lease identities; no all-transient-PID claim",
        "resource_close": close, "owned_output_bytes_before_review": total,
        "not_run": ["remaining_supplemental_generic", "three_fresh_interleaved_formal_pairs", "external_trained_gaussian",
                    "new_training_or_GT", "production_or_existing_preview_deployment"],
        "review_seconds_before_publication": time.perf_counter() - started,
        "review_pid": os.getpid(), "review_creation_time": psutil.Process().create_time(),
        "actual_reviewer_self_and_parents": ancestry, "review_lease": str(lease_receipt),
        "review_process_and_OS_lock_post_exit_check": "pending_after_this_review_process_exits",
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: payload[name] for name in ("status", "accepted_count_contribution", "complete_steps_each",
          "actual_full_byte_comparisons", "actual_science_files", "screen_raw_complete_reduction", "recorded_owned_instances_closed")}))


if __name__ == "__main__":
    main()
