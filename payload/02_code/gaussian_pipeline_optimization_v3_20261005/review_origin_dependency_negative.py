"""Review preserved GPU runtime failure and closure without another browser run."""

import argparse
import hashlib
import json
from pathlib import Path

import psutil


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gone(pid, created):
    try:
        return psutil.Process(pid).create_time() != created
    except psutil.NoSuchProcess:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    b = args.campaign
    final = read(b / "FINAL_RESULT.json")
    failure = read(b / "lifecycle_002/GENERIC_FAILURE.json")
    raw_result = read(b / "lifecycle_002/RESULT.json")
    primary = read(b / "generic_003/GENERIC_PRIMARY_RESULT.json")
    close = read(b / "CLOSURE.json")
    drain = read(b / "POST_CLOSE_PID_RECEIPT.json")
    complete = read(b / "GOAL_COMPLETION.json")
    native = read(b / "lifecycle_002/actual/native_scene_real_context_restore_rotate/STATE.json")
    checks = {
        "actual_candidate_guard_failure_zero": final["status"] == "failed_verification"
        and final["classification"] == "actual_candidate_runtime_guard_failure"
        and final["contribution"] == 0 and final["counted"] is False,
        "raw_automatic_classification_preserved": raw_result["status"] == failure["status"]
        == "validator_setup_failed",
        "actual_failure_case_and_GPU_error_preserved": failure["current"]["name"]
        == "scene_real_context_restore_rotate" and "Actual GPU completion reported an error" in failure["error"],
        "native_same_restore_capture_succeeded": native["pointCount"] == 526951
        and native["numSh"] == 3 and native["errors"] == []
        and native["sourceSortingClosed"] is True and native["trace"]["contextLossRestoration"] == 1,
        "candidate_failed_frame_not_falsely_complete": final["first_required_failure"]["candidate_capture_completed"] is False
        and not (b / "lifecycle_002/actual/candidate_scene_real_context_restore_rotate/STATE.json").exists(),
        "numeric_code_and_byte_mismatch_root_cause_not_invented": all(
            final["first_required_failure"][k] is False for k in
            ("GL_error_numeric_code_saved", "byte_mismatch_claimed", "exact_root_cause_confirmed")),
        "no_performance_or_whole_acceptance": final["cost"]["formal_pairs_completed"] == 0
        and final["cost"]["phase_reduction"] is None and final["cost"]["whole_cost_effect"] is None,
        "local118_only_partial_acceptance": len(primary["completed"]) == final["primary"]["paired_cases"] == 118
        and primary["fullSemanticAcceptance"] is False and final["primary"]["full_semantic_acceptance"] is False,
        "all_required_conditionals_stopped": all(v.startswith("not_run") for v in final["not_run"].values()),
        "real_own_goal_complete_and_parent_unchanged": complete["goal"]["status"] == "complete"
        and complete["contribution"] == 0 and final["parent_goal_closed"] is False,
        "seven_leases_fourteen_recorded_identities": drain["lease_count"] == len(drain["receipts"]) == 7
        and drain["process_identity_count"] == len(drain["processes"]) == 14,
        "all_current_recorded_PID_creation_identities_gone": all(
            gone(r["pid"], r["recorded_creation_time"]) for r in drain["processes"]),
        "all_original_leases_released": all(read(Path(r["path"]))["lease_released_utc"] == r["released_utc"]
            for r in drain["receipts"]),
        "final_bindings_workers_GPU_close_receipts": close["final_context_bindings_zero"]
        and close["final_worker_observers_restored"] and close["final_actual_GPU_drains_passed"],
        "older_native_cleanup_failure_not_erased": close["old_setup_cleanup_failure_preserved"]
        == "lifecycle_001 native GPU drain failed; its private page/browser was destroyed and both controller PIDs are closed"
        and close["nodes"]["lifecycle_001"]["original_cleanup_rows"][0]["completeRestoration"] is False
        and "Actual GPU completion reported an error" in close["nodes"]["lifecycle_001"]["original_cleanup_rows"][0]["error"],
        "Chromium_leaf_ledger_limit_disclosed": close["browser_PID_ledger_not_captured"] is True,
        "all_owned_browser_server_close_receipts": all(n["browser_closed_receipt"]
            and n["server_closed_receipt"] for n in close["nodes"].values()),
    }
    asset_count = 0
    asset_bytes = 0
    all_sizes = True
    all_saved_comparisons = True
    all_camera_and_pixels_sizes = True
    for row in primary["completed"]:
        all_saved_comparisons &= all(v["completeBytesEqual"] is True for v in row["check"].values())
        for label, state in row["rows"].items():
            directory = b / "generic_003/actual" / state["label"]
            all_sizes &= (directory / "STATE.json").is_file()
            asset_count += 1
            for name, item in row["check"].items():
                path = directory / name
                all_sizes &= path.is_file() and path.stat().st_size == item["bytes"]
                asset_count += 1
                asset_bytes += item["bytes"]
            all_camera_and_pixels_sizes &= row["check"]["camera.bin"]["bytes"] == 312
            all_camera_and_pixels_sizes &= row["check"]["pixels.bin"]["bytes"] == state["width"] * state["height"] * 4
    checks["all118_saved_complete_byte_comparisons_recorded_true"] = all_saved_comparisons
    checks["all1432_primary_actual_files_and_all_recorded_sizes_exist"] = all_sizes and asset_count == 1432
    checks["all_primary_camera_and_complete_framebuffer_sizes"] = all_camera_and_pixels_sizes
    native_dir = b / "lifecycle_002/actual/native_scene_real_context_restore_rotate"
    checks["required_native_complete_failure_comparator_files_exist"] = all(
        (native_dir / name).is_file() and (native_dir / name).stat().st_size > 0
        for name in ("pixels.bin", "ordering_gpu.bin", "ordering_worker.bin", "packed_gpu.bin", "camera.bin"))
    candidate = final["candidate"]
    checks["actual_frozen_candidate_source_identity_preserved"] = (
        Path(candidate["path"]).stat().st_size == candidate["bytes"]
        and hashlib.sha256(Path(candidate["path"]).read_bytes()).hexdigest() == candidate["sha256"])
    checks["all85_frozen_source_copies_saved_equal_and_exist"] = len(close["source_snapshots"]) == 85 and all(
        r["complete_source_bytes_equal"] is True and Path(r["path"]).is_file() for r in close["source_snapshots"])
    small = [r for r in close["identities"] if r["bytes"] < 1024**2
        and Path(r["path"]).suffix in (".py", ".js", ".mjs", ".cjs", ".bin", ".spk")]
    checks["current_small_source_input_identities_actually_preserved"] = all(
        Path(r["path"]).stat().st_size == r["bytes"]
        and hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["sha256"] for r in small)
    report = {"status": "negative_scope_review_passed" if all(checks.values()) else "review_failed",
        "candidate": "R115", "count_contribution": 0, "checks": checks, "check_count": len(checks),
        "primary_actual_file_sizes_checked": asset_count, "primary_recorded_binary_bytes": asset_bytes,
        "small_source_input_full_identities_rechecked": len(small),
        "current_owned_pid_creation_identities_rechecked": 14,
        "large_image_model_input_and1432_GPU_byte_bodies_recompared_by_root": False,
        "source_root_cause_or_complete_semantic_performance_acceptance": False,
        "reviewer_new_browser_GPU_or_training_runs": 0,
        "numeric_GL_code_not_saved_and_old_failure_preserved": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
        "failed_checks": [k for k, v in checks.items() if not v], "primary_files": asset_count}, allow_nan=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
