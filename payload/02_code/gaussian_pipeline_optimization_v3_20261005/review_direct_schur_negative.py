"""Independently check the saved cost failure without solving another map."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import psutil


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close_seconds(lease):
    end = datetime.datetime.fromisoformat(lease["lease_released_utc"]).timestamp()
    return end - lease["guard_creation_time"] - lease["wait_seconds"]


def is_gone(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = args.campaign
    prereg = read(base / "PREREGISTRATION.json")
    campaign = read(base / "CAMPAIGN_RESULT.json")
    final = read(base / "R114_FINAL_RESULT.json")
    sides = read(base / "CAMPAIGN_SIDES.json")
    checks = {
        "fixed_whole_guard": prereg["gates"]["each_whole_direct_over_auto_max"] == 1.1,
        "fixed_first_failure_rule": prereg["first_actual_failure_stops_dependents"] is True,
        "actual_pair1_cost_failure": campaign["failure"]["node"] == "pair1"
        and campaign["failure"]["phase"] == "each_whole_cost_guard",
        "zero_contribution": final["effective_count"] == 0 and campaign["counted"] is False,
        "four_completed_nodes": len(sides) == campaign["completed_node_count"] == 4,
        "exact_frozen_order": [side["node"] for side in sides]
        == ["pair0_auto", "pair0_direct", "pair1_direct", "pair1_auto"],
        "pair2_not_created": not any((base / n).exists() for n in ("pair2_auto", "pair2_direct")),
        "all_actual_dependents_not_run": all(v == "not_run_after_failure"
        for k, v in campaign["dependent_status"].items() if k != "root_independent_review"),
        "root_review_pending_accurately_corrected": final["root_independent_review"] == "pending_root_review",
        "first_failed_oracle_preserved": (base / "oracle_c001/FAILURE.json").is_file()
        and (base / "oracle_c001/profile_source_before_repair.py").is_file()
        and (base / "oracle_c001/checker_source_before_repair.py").is_file(),
    }
    identities = {}
    small_sources = {}
    costs = {}
    setup = sum(close_seconds(read(base / (name + "_LEASE.json")))
        for name in ("backend_c001", "oracle_c001", "oracle_c002"))
    checks["setup_original_clocks_exact"] = setup == campaign["setup_closed_seconds"]
    all_contracts = []
    native_solves = []
    for side in sides:
        name = side["node"]
        lease = read(base / (name + "_LEASE.json"))
        resource = read(base / (name + "_RESOURCE.json"))
        contract = read(base / name / "CONTRACT.json")
        all_contracts.append(contract)
        seconds = close_seconds(lease)
        assigned = setup if name == "pair0_direct" else 0.0
        costs[name] = seconds + assigned
        checks[name + "_original_clock_exact"] = seconds == side["closed_node_seconds"]
        checks[name + "_phase_whole_charges_exact"] = costs[name] == side["whole_seconds"] == side["completephase_seconds"]
        checks[name + "_saved_resource_limits_and_completion"] = (
            resource["status"] == "completed" and resource["returncode"] == 0
            and resource["maximum_seconds"] == 1800
            and resource["maximum_RSS_bytes"] == 8 * 1024**3
            and resource["peak_tree_RSS_bytes"] < 8 * 1024**3
            and side["result"]["status"] == "metric_complete_passed")
        checks[name + "_original_input_counts"] = contract["input_counts"] == {
            "cameras": 1, "registered_images": 1021, "points3D": 75846, "observations": 959881}
        checks[name + "_saved_restore_source_guards"] = (
            side["result"]["interface_restored"] is True
            and side["result"]["sources_inputs_unchanged"] is True
            and all(read(base / name / "INTERFACE_RESTORE.json").values())
            and all(read(base / name / "GUARDS.json").values()))
        for ident in resource.get("owned_process_identities", []) + [
            {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]:
            identities[(ident["pid"], ident["creation_time"])] = ident
        for item in contract["source"]:
            path = Path(item["path"])
            if path.suffix == ".py" and item["bytes"] < 1024**2:
                small_sources[str(path)] = item
        for label in ("visual_metric_aligned", "weak_position_prior"):
            readback = read(base / name / (label + "_FULL_READBACK.json"))
            checks[name + "_" + label + "_saved_full_readback_guards"] = all(readback["checks"].values())
            item = side["result"]["original_result"]["results"][label]
            native = item["solver"]["native_summary"]["ceres_summary"]
            expected = "SPARSE_SCHUR" if side["mode"] == "direct" else "ITERATIVE_SCHUR"
            checks[name + "_" + label + "_actual_backend"] = (
                native["linear_solver_type_used"] == expected
                and native["num_threads_given"] == native["num_threads_used"] == 8
                and native["num_linear_solves"] > 0
                and native["linear_solver_time_in_seconds"] > 0
                and (side["mode"] != "direct" or native["sparse_linear_algebra_library_type"] == "SUITE_SPARSE"))
            native_solves.append({"node": name, "case": label,
                "used": native["linear_solver_type_used"], "termination": item["solver"]["termination_type"],
                "linear_seconds": native["linear_solver_time_in_seconds"]})
    checks["all_four_saved_complete_sources_and_inputs_equal"] = all(
        c["source"] == all_contracts[0]["source"] and c["inputs"] == all_contracts[0]["inputs"]
        for c in all_contracts)
    checks["all_small_actual_source_bytes_preserved"] = all(
        Path(path).stat().st_size == ident["bytes"]
        and hashlib.sha256(Path(path).read_bytes()).hexdigest() == ident["sha256"]
        for path, ident in small_sources.items())
    for name in ("backend_c001", "oracle_c001", "oracle_c002", "closure_c001"):
        lease = read(base / (name + "_LEASE.json"))
        resource = read(base / (name + "_RESOURCE.json"))
        checks[name + "_lease_released"] = lease["status"] == "child_exited" and bool(lease["lease_released_utc"])
        for ident in resource.get("owned_process_identities", []) + [
            {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]:
            identities[(ident["pid"], ident["creation_time"])] = ident
    ratio = costs["pair1_direct"] / costs["pair1_auto"]
    checks["pair1_independent_ratio_exact_and_failed"] = (
        ratio == campaign["failure"]["observed_ratio"]
        == final["first_actual_required_failure"]["observed"] and ratio > 1.1)
    parent = final["costs_seconds"]["parent_dispatch_actual_close_bookkeeping"]
    sensitivity = costs["pair1_direct"] / (costs["pair1_auto"] + parent)
    checks["full_parent_close_allowance_cannot_rescue_failure"] = sensitivity > 1.1
    total = (sum(side["closed_node_seconds"] for side in sides) + setup + parent
        + close_seconds(read(base / "closure_c001_LEASE.json")))
    checks["total_measurement_and_closure_rederived"] = abs(total - final["costs_seconds"]["total_measured_and_closed"]) < 1e-9
    checks["all_current_owned_PID_creation_identities_gone"] = all(is_gone(i) for i in identities.values())
    report = {"status": "negative_scope_review_passed" if all(checks.values()) else "review_failed",
        "candidate": "R114", "count_contribution": 0, "checks": checks,
        "check_count": len(checks), "native_solves": native_solves,
        "independently_recomputed_pair1_ratio": ratio,
        "favourable_all_parent_close_allowance_ratio": sensitivity,
        "independently_recomputed_total_seconds": total,
        "current_owned_PID_creation_identities_checked": len(identities),
        "actual_small_source_bytes_checked": len(small_sources),
        "large_native_binary_input_and_model_bytes_reloaded_by_reviewer": False,
        "scientific_acceptance_or_weak_pose_accuracy_claim": False,
        "reviewer_started_new_solver_training_or_performance_run": False,
        "original_autogenerated_root_review_not_run_is_corrected_by_final": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
        "failed_checks": [k for k, v in checks.items() if not v], "pair1_ratio": ratio,
        "owned_identities": len(identities)}, allow_nan=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
