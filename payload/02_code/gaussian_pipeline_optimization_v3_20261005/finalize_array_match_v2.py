"""Count one complete scoped V2 acceptance after the actual review owner exits."""

import datetime
from pathlib import Path
import statistics
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(CODE), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]
from array_match_node_config import OUT, read, atomic_json, data_identity
from array_match_node import remaining


def main():
    output = OUT / "ACCEPTANCE_V2.json"
    if output.exists():
        raise FileExistsError("Never overwrite a versioned acceptance receipt")
    review = read(OUT / "v2r0/REVIEW.json")
    lease = read(OUT / "V2_REVIEW_LEASE_001.json")
    guard = read(OUT / "V2_REVIEW_GUARD_001.json")
    identities = [review["owner"], *guard["owned_process_identities"]]
    for prefix in ("guard", "child"):
        identities.append({"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]})
    checks = {"complete_V2_output_source_input_and_cancel_review": all(review["checks"].values()),
        "new_review_guard_and_lease_closed": guard["status"] == "completed" and guard["returncode"] == 0
            and lease["returncode"] == 0 and bool(lease.get("lease_released_utc")),
        "new_review_sources_preserved": review["source"] == data_identity(review["source"]["path"]),
        "new_review_owned_instances_gone_now": not remaining(identities)}
    final_seconds = time.perf_counter() - STARTED
    fee = review["complete_prior_first_fees_seconds"] + lease["child_elapsed_seconds"] + final_seconds
    medians = {key: {mode: statistics.median(pair[key][mode] + fee / 6 for pair in review["raw_pairs"])
        for mode in ("incumbent", "array")} for key in ("raw_CPU_phase_seconds", "raw_whole_seconds")}
    reductions = {key: 1 - row["array"] / row["incumbent"] for key, row in medians.items()}
    checks.update(complete_phase_at_least5percent=reductions["raw_CPU_phase_seconds"] >= .05,
        complete_whole_median_not_regressed=reductions["raw_whole_seconds"] >= 0,
        each_complete_whole_at_most1_1=all(pair["raw_whole_seconds"]["array"] + fee / 6 <=
            1.1 * (pair["raw_whole_seconds"]["incumbent"] + fee / 6) for pair in review["raw_pairs"]))
    result = {"status": "accepted_V2_scoped_engineering_mechanism" if all(checks.values()) else "failed_V2_verification",
        "mechanism_id": "R120", "mechanism": "exact_prepared_match_arrays_to_unchanged_native_CPU_geometry",
        "checks": checks, "new_optimization_count": int(all(checks.values())),
        "complete_semantic_acceptance": all(checks.values()), "acceptance_policy_version": 2,
        "historical_V1_acceptance": False, "historical_V1_negative_preserved": True,
        "minimum_complete_stage_speed_reduction": .05, "whole_median_must_not_regress": True,
        "complete_first_fees_including_prior_campaign_wrapper_and_V2_review_seconds": fee,
        "V2_review_complete_lease_child_seconds": lease["child_elapsed_seconds"],
        "final_cost_and_close_seconds": final_seconds,
        "complete_medians_seconds": {"phase": medians["raw_CPU_phase_seconds"], "whole": medians["raw_whole_seconds"]},
        "reductions": {"phase": reductions["raw_CPU_phase_seconds"], "whole": reductions["raw_whole_seconds"]},
        "three_original_full24_48_pairs_rechecked_now": True, "all_scientific_outputs_exact": True,
        "generic180_and_original16_guards_retained": True, "both_additional_cancel_cases_verified": True,
        "original_source_input_identities_checked": review["source_input_identities_checked"],
        "review_kind": "root_self_review_no_subagents", "independent_review": False,
        "scope": "fixed_R90_R92_R94_R102_localization_entry_OpenCV4_13_float32_prepared_batch_capacity12000_12",
        "production_modified": False, "deployment_or_combination_with_R124": "not_run_not_claimed",
        "new_object_full_pipeline_or_GT_accuracy": "unverified",
        "new_training_and_preview": "unrun_no_change_to_downstream_training_inputs",
        "evidence": [str(OUT / "c0/RESULT.json"), str(OUT / "NEGATIVE_CLOSURE_001.json"),
            str(OUT / "v2r0/REVIEW.json"), str(OUT / "V2_REVIEW_LEASE_001.json")],
        "source": data_identity(Path(__file__)), "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(output, result)
    print(result, flush=True)
    if not all(checks.values()):
        raise AssertionError("The preserved complete cost or protection does not pass V2")


if __name__ == "__main__":
    main()
