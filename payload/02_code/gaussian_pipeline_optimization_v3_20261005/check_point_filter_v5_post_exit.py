"""Validate closed publisher/finalizer identities and the reserved V5 cost tail."""

import argparse
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from close_point_filter_v5 import OWN, BASE, DOC, read, released, costs, closed


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    (output / "EXECUTED_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    candidate = read(OWN / "ACCEPTANCE_V5_001.json")
    review = read(OWN / "r0/RESULT.json")
    state = read(DOC / "CURRENT_STATE.json")
    checks, all_instances = {}, list(review["actual_saved_closed_process_instances"])
    for label in ("R0", "P0", "F0"):
        lease, identities = released(label)
        all_instances.extend(identities)
        checks[label + "_actually_exited_and_OS_lease_released"] = True
    original = read(BASE / "point_filter_predicate_001/campaign_001/RESULT.json")
    recomputed = costs(original, candidate["known_all_metered_fees_before_finalizer_seconds"] + 10)
    checks["complete_original_six_costs_medians_each_ratio_and_3percent_lower_bound_exact"] = recomputed == candidate["costs"] and recomputed["complete_phase_reduction_lower_bound"] >= .03 and recomputed["complete_whole_median_ratio"] <= 1.01 and max(recomputed["each_whole_ratio"]) <= 1.1
    f0 = read(OWN / "LEASE_F0.json")
    f0_guard = read(OWN / "GUARD_F0.json")
    elapsed = time.perf_counter()-started
    checks["remaining10second_reserve_kept_and_finalizer_guard_within5seconds2GiB"] = candidate["reserved_remaining_measured_execution_seconds"] == 10 and f0_guard["maximum_seconds"] == 5 and f0_guard["maximum_RSS_bytes"] == 2*1024**3 and f0_guard["elapsed_seconds"] <= 5 and f0["child_elapsed_seconds"] + elapsed < 10
    checks["current33_semantic_guards_and27_unique_receipts_remain_protected"] = len(review["checks"]) == 33 and all(review["checks"].values()) and state["additional_accepted_count"] == len(set(state["accepted_mechanisms"])) == len(set(state["acceptance_receipts"])) == 27 and not any(name.startswith("R113") for name in state["accepted_mechanisms"])
    checks["current_review_candidate_and_publisher_code_frozen"] = data_identity(Path(__file__).with_name("review_point_filter_v5.py")) == review["V5_review_source"] and data_identity(Path(__file__).with_name("point_filter_predicate.py")) == review["original_candidate_source"] and data_identity(Path(__file__).with_name("close_point_filter_v5.py")) == candidate["publisher_source"]
    checks["old_strict_and_V4_failure_artifacts_preserved0"] = read(BASE / "point_filter_predicate_001/campaign_001/RESULT.json")["status"] == "complete_filter_cost_guard_failed" and read(BASE / "pf_v4_001/V4_NEGATIVE_CLOSURE_001.json")["new_optimization_count"] == 0
    prior = BASE / "native_quality_cache_writer_001"
    native_result = read(prior / "r1/RESULT.json")
    checks["R151_negative_actual_complete128_and22_checks_kept0"] = native_result["accepted_count_contribution"] == 0 and len(native_result["checks"]) == 22 and all(native_result["checks"].values()) and native_result["reduction"] < 0
    for label in ("B0", "B1", "G0", "G1", "S0", "R0", "R1"):
        lease = read(prior / ("LEASE_" + label + ".json"))
        guard = read(prior / ("GUARD_" + label + ".json"))
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
            raise ValueError("An earlier negative cache task retained its lease")
        all_instances.extend(guard["owned_process_identities"])
        all_instances.extend({"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]} for role in ("guard", "child"))
    unique = {(row["pid"], row["creation_time"]): row for row in all_instances}
    checks["all_saved_earlier_positive_and_negative_actual_PID_ctime_instances_exited"] = all(closed(row) for row in unique.values())
    links = [OWN / "r0/RESULT.json", OWN / "h8/RESULT.json", BASE / "pf_v4_001/v5_e0/RESULT.json",
             BASE / "point_filter_predicate_001/v5_delete_001/RESULT.json", DOC / "ENGINEERING_ACCEPTANCE_V5.md",
             DOC / "R113_V5_OPTIONAL_FILTER_TASK.md", DOC / "R151_NATIVE_QUALITY_CACHE_RESULTS.md"]
    checks["all_current_evidence_targets_exist_native_text_JSON_parsed"] = all(path.is_file() for path in links) and all(isinstance(read(path), dict) for path in links if path.suffix == ".json")
    if not all(checks.values()):
        atomic_json(output / "FAILURE.json", {"checks": checks, "new_optimization_count": 0})
        raise ValueError("A final exit, reserve, source or frontier check failed")
    atomic_json(output / "RESULT.json", {"status": "actual_prior_exits_sources_full_cost_lower_bound_and_reserve_passed_own_exit_still_pending",
        "checks": checks, "all_saved_earlier_instances_exited": list(unique.values()), "reserved_final_envelope_seconds": 10,
        "actual_finalizer_closed_envelope_seconds": f0["child_elapsed_seconds"], "current_node_internal_seconds": time.perf_counter()-started,
        "conservative_costs": recomputed, "new_optimization_count": 0, "parent27_until_this_node_actual_exit_verified": True,
        "root_self_review_not_independent_agent": True, "no_production_training_GT_or_unseen_full_chain_claim": True})
    print({"status": "post_exit_checks_passed_own_exit_pending", "checks": len(checks), "prior_instances_exited": len(unique)}, flush=True)


if __name__ == "__main__":
    main()
