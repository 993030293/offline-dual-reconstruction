"""Close actual measured owners and apply the preregistered five-percent policy."""

import time
STARTED = time.perf_counter()

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import psutil

from profile_sparse_ply_oracle import ROOT
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_process(identity):
    try:
        process = psutil.Process(identity["pid"])
        return process.create_time() == identity["creation_time"] and process.is_running()
    except psutil.NoSuchProcess:
        return False


def main():
    targets = [OUT / name for name in ("CLOSURE_VERIFICATION_001.json", "FINAL_COST_001.json", "ACCEPTANCE.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("A final acceptance closure must use fresh output identities")
    review = read(OUT / "r0/REVIEW.json")
    review_guard = read(OUT / "REVIEW_GUARD_001.json")
    review_lease = read(OUT / "REVIEW_LEASE_001.json")
    owner = read(OUT.parent / "coordination_001/LOCK_OWNER.json")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    campaign = read(OUT / "c1/RESULT.json")
    source = data_identity(Path(__file__))
    own = {(row["pid"], row["creation_time"]): row for row in review_guard["owned_process_identities"]}
    for prefix in ("guard", "child"):
        row = {"pid": review_lease[prefix + "_pid"], "creation_time": review_lease[prefix + "_creation_time"]}
        own[(row["pid"], row["creation_time"])] = row
    all_instances = {(row["pid"], row["creation_time"]): row
                     for row in review["Windows_process_identities_checked"]}
    all_instances.update(own)
    live = [row for row in all_instances.values() if same_process(row)]
    prior_costs = review["root_costs_without_nested_double_counting"]
    lease_rows = [read(Path(row["path"])) for row in prior_costs]
    checks = {
        "current_root_self_review21_passed": review["status"] == "root_review_passed_final_complete_cost_and_own_close_pending"
            and len(review["checks"]) == 21 and all(review["checks"].values()),
        "review_actual_child_completed_normally": review_guard["status"] == "completed" and review_guard["returncode"] == 0
            and review_lease["status"] == "child_exited" and review_lease["returncode"] == 0,
        "all_prior_and_review_actual_processes_gone": not live,
        "all_fifteen_measured_leases_released": len(lease_rows) == 14
            and all(row["status"] == "child_exited" and row.get("lease_released_utc") for row in lease_rows)
            and bool(review_lease["lease_released_utc"]) and owner["status"] == "released"
            and owner["owner"] == review_lease["owner"],
        "parent17_not_already_counting_this_candidate": state["additional_accepted_count"] == 17
            and len(set(state["accepted_mechanisms"])) == 17 and len(state["acceptance_receipts"]) == 17
            and "R132_WHOLE_MAP_SEGMENTED_CAMERA_SUPPORT" not in state["accepted_mechanisms"],
        "original_v2_5percent_and_all_quality_protections_unchanged": state["minimum_complete_stage_speed_reduction_current"] == .05
            and state["quality_geometry_output_recovery_and_whole_gates_unchanged"],
        "three_formal_fresh_pairs_alternating_order_retained": [row["order"] for row in campaign["pairs"]]
            == [["native", "segmented"], ["segmented", "native"], ["native", "segmented"]],
    }
    closure = {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "review": str(OUT / "r0/REVIEW.json"), "review_kind": review["review_kind"],
        "actual_prior_Windows_instances_checked": len(review["Windows_process_identities_checked"]),
        "actual_review_instances_checked": len(own), "actual_deduplicated_closed_instances": len(all_instances),
        "live_instances": live, "measured_leases_closed": 15,
        "read_only_closure_observer_is_not_a_training_or_benchmark_worker": True,
        "closure_source": source, "new_optimization_count": 0,
    }
    atomic_json(targets[0], closure)
    if not all(checks.values()):
        raise AssertionError("A final measured owner or unchanged acceptance protection did not close")
    first_fees = review["complete_prior_first_failure_diagnostic_lifecycle_and_external_fees_seconds"]
    overhead = review["current_campaign_setup_compare_publish_and_close_seconds"]
    review_fee = review_lease["child_elapsed_seconds"]
    archive_fees = [
        {"scope": "source_history_001_copy_and_receipt_shell_tool_wall", "seconds": .2006523},
        {"scope": "source_history_002_copy_and_receipt_shell_tool_wall", "seconds": .2032429},
    ]
    # The recorded archival shell envelopes conservatively include their startup.
    closure_observation_fee = time.perf_counter() - STARTED
    fee_total = first_fees + overhead + review_fee + sum(row["seconds"] for row in archive_fees) + closure_observation_fee
    shared = fee_total / 6
    final_pairs = []
    for row in campaign["pairs"]:
        nodes = {mode: {scope: row["nodes"][mode][scope] + shared
                        for scope in ("complete_phase_seconds", "complete_whole_seconds")}
                 for mode in ("native", "segmented")}
        final_pairs.append({"pair": row["pair"], "order": row["order"], "nodes": nodes,
                            "complete_whole_ratio": nodes["segmented"]["complete_whole_seconds"] / nodes["native"]["complete_whole_seconds"],
                            "original_raw_whole_ratio": row["whole_ratio"]})
    medians = {scope: {mode: statistics.median(row["nodes"][mode][scope] for row in final_pairs)
                       for mode in ("native", "segmented")}
               for scope in ("complete_phase_seconds", "complete_whole_seconds")}
    reductions = {scope: 1 - values["segmented"] / values["native"] for scope, values in medians.items()}
    gates = {
        "complete_phase_median_at_least_five_percent_less": reductions["complete_phase_seconds"] >= .05,
        "complete_whole_median_not_increased": reductions["complete_whole_seconds"] >= 0,
        "all_complete_and_original_per_pair_whole_ratios_at_most1_1": all(row["complete_whole_ratio"] <= 1.1
            and row["original_raw_whole_ratio"] <= 1.1 for row in final_pairs),
        "all_prior_required_source_output_lifecycle_external_and_resource_checks_passed": all(review["checks"].values()) and all(checks.values()),
        "closure_source_unchanged": source == data_identity(Path(__file__)),
    }
    cost = {
        "status": "complete_measured_cost_passed" if all(gates.values()) else "failed_complete_cost",
        "policy": state["engineering_acceptance_policy"], "gates": gates,
        "raw_three_pair_medians": campaign["medians_before_outer_first_and_review_fees"],
        "all_prior_first_failure_diagnostic_generic_lifecycle_external_lease_costs": prior_costs,
        "prior_first_fees_seconds": first_fees, "current_campaign_overhead_seconds": overhead,
        "complete_review_lease_seconds_including_publication_and_process_close": review_fee,
        "recorded_archival_tool_envelopes": archive_fees, "closure_observation_through_check_publication_seconds": closure_observation_fee,
        "all_shared_measured_fees_seconds": fee_total, "shared_fees_per_side_seconds": shared,
        "complete_pairs": final_pairs, "complete_medians": medians, "complete_reductions": reductions,
        "measured_runtime_scope": "all required actual algorithm and validation lease nodes, full campaign comparisons, first failures, review shutdown, archival tool envelopes, and final read-only closure observation",
        "outside_algorithm_runtime": "source authoring, knowledge/manual text lookup, report writing, and the final small acceptance-record publication after this read-only closure observation",
        "formal_pairs_restarted_or_slow_pair_removed": False,
        "source": source,
    }
    atomic_json(targets[1], cost)
    acceptance = {
        "status": "accepted_scoped_engineering_mechanism" if all(gates.values()) else "failed_verification",
        "mechanism": "R132_WHOLE_MAP_SEGMENTED_CAMERA_SUPPORT", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "new_optimization_count": 1 if all(gates.values()) else 0, "review_kind": review["review_kind"],
        "policy": state["engineering_acceptance_policy"], "gates": gates,
        "complete_phase_reduction": reductions["complete_phase_seconds"],
        "complete_whole_reduction": reductions["complete_whole_seconds"], "complete_medians": medians,
        "review": str(OUT / "r0/REVIEW.json"), "cost": str(targets[1]), "resource_closure": str(targets[0]),
        "current_source_input_identities_verified": review["unique_current_source_input_identities"],
        "current_and_historical_source_copies_verified": review["saved_source_copies"],
        "local_effect_confirmed": "three full original exports, all1050 artifacts and three3053 support states exact",
        "external_component_confirmed": "fixed existing Oxford360 component, three exact complete camera-field pairs",
        "generalization_pending": "complete previously unseen object pipeline, quality and independent metric GT",
        "deployment": "isolated versioned code and experiments; production default not changed",
        "new_complete_training_or_preview": "not run because all original training inputs and map artifacts are byte identical",
        "parent_target": 50, "expected_parent_count_after_separate_state_update": 18,
        "native_parent_goal_completed": False,
    }
    atomic_json(targets[2], acceptance)
    print(json.dumps({key: acceptance[key] for key in ("status", "new_optimization_count", "complete_phase_reduction", "complete_whole_reduction", "complete_medians")}))
    if not all(gates.values()):
        raise AssertionError("The preregistered complete cost acceptance gates failed")


if __name__ == "__main__":
    main()
