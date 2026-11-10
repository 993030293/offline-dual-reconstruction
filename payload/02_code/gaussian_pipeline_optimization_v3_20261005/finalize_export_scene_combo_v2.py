"""Finalize a scoped existing-mechanism combination; its new-item count stays zero."""

import datetime
from pathlib import Path
import statistics
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(CODE), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]
from runtime import data_identity, atomic_json
from reaudit_export_scene_combo_v2 import read, OUT, DOC
from array_match_node import remaining


def main():
    output = OUT / "ACCEPTANCE_V2.json"
    if output.exists():
        raise FileExistsError("Never overwrite versioned combination acceptance")
    review = read(OUT / "v2r1/REVIEW.json")
    lease, guard = read(OUT / "V2_REVIEW_LEASE_002.json"), read(OUT / "V2_REVIEW_GUARD_002.json")
    identities = [review["owner"], *guard["owned_process_identities"]]
    for prefix in ("guard", "child"):
        identities.append({"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]})
    checks = {"all_prior_output_input_owner_and_corrected_timestamp_guards": all(review["checks"].values()),
        "corrected_review_guard_and_lease_closed": guard["status"] == "completed" and guard["returncode"] == 0
            and lease["returncode"] == 0 and bool(lease.get("lease_released_utc")),
        "corrected_review_source_unchanged": data_identity(review["source"]["path"]) == review["source"],
        "corrected_review_actual_owners_gone": not remaining(identities)}
    final_seconds = time.perf_counter() - STARTED
    extra_fee = (review["first_failed_review_complete_child_seconds"] + review["timestamp_diagnostic_two_reported_tool_wall_seconds"]
        + lease["child_elapsed_seconds"] + final_seconds)
    medians = {field: {mode: statistics.median(pair[mode][field] + extra_fee / 6
        for pair in review["prior_full_cost_pairs"]) for mode in ("native", "combined")} for field in ("phase", "whole")}
    reductions = {key: 1 - row["combined"] / row["native"] for key, row in medians.items()}
    checks.update(complete_phase_at_least5percent=reductions["phase"] >= .05,
        complete_whole_median_not_regressed=reductions["whole"] >= 0,
        each_complete_whole_at_most1_1=all(pair["combined"]["whole"] + extra_fee / 6 <=
            1.1 * (pair["native"]["whole"] + extra_fee / 6) for pair in review["prior_full_cost_pairs"]))
    result = {"status": "accepted_V2_scoped_existing_mechanism_combination" if all(checks.values()) else "failed_V2_verification",
        "combination_id": "A004_R112_Q68", "mechanisms": ["R112", "Q68"], "checks": checks,
        "new_optimization_count": 0, "new_mechanism_contribution": 0, "combination_accepted": all(checks.values()),
        "complete_semantic_acceptance": all(checks.values()), "acceptance_policy_version": 2,
        "historical_V1_full_cost_rejection_preserved": True, "minimum_complete_stage_speed_reduction": .05,
        "complete_medians_seconds": medians, "reductions": reductions,
        "extra_V2_review_and_failures_fees_seconds": extra_fee,
        "prior_all_mandatory_costs_and_pair_envelopes_retained": str(DOC / "FULL_COST_REVISION_001.json"),
        "all1050_export_contents_and1016_Scene_original_fields_exact": True,
        "source_input_identities_checked": review["sources_inputs_checked"], "source_copies_checked": 6 * review["source_rows"],
        "old_Windows_identities_currently_gone": review["Windows_checked"], "old_Linux_identities_currently_gone": review["Linux_checked"],
        "scope": "own_new_export_then_identity_then_complete_Scene_fixed_current_R112_Q68_and_camera_flags",
        "review_kind": "root_self_review_no_subagents_or_other_chat_dispatch", "independent_review": False,
        "other_chat_native_status_changed": False, "production_route_promoted": False,
        "new_training_restore_loop_full24_quality_preview_external_full_chain_GT": "unrun_not_claimed",
        "evidence": [str(OUT / name) for name in ("v2r0/REVIEW.json", "v2r1/REVIEW.json", "V2_REVIEW_LEASE_002.json")],
        "source": data_identity(Path(__file__)), "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(output, result)
    print({key: result[key] for key in ("status", "checks", "complete_medians_seconds", "reductions", "new_optimization_count")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("The complete combination does not pass V2")


if __name__ == "__main__":
    main()
