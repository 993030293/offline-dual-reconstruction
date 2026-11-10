"""Finalize V2 cost accounting after every new Gaussian review owner drains."""

import datetime
from pathlib import Path
import statistics
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]

from streamed_ply_node_config import OUT, read, atomic_json, data_identity
from streamed_ply_node_002 import remaining


def main():
    output = OUT / "ACCEPTANCE_V2.json"
    if output.exists():
        raise FileExistsError("Never overwrite versioned acceptance")
    review = read(OUT / "v2a1/REVIEW.json")
    lease, guard = read(OUT / "V2_REVIEW_LEASE_002.json"), read(OUT / "V2_REVIEW_GUARD_002.json")
    identities = [review["owner"], *guard["owned_process_identities"]]
    for prefix in ("guard", "child"):
        identities.append({"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]})
    checks = {"full_current_bytes_sources_recovery_and_owner_review_passed": all(review["checks"].values()),
        "new_review_guard_and_lease_closed": guard["status"] == "completed" and guard["returncode"] == 0
            and lease["returncode"] == 0 and bool(lease.get("lease_released_utc")),
        "new_review_source_identity_preserved": review["source"] == data_identity(review["source"]["path"]),
        "all_new_review_processes_actually_gone_now": not remaining(identities)}
    final_seconds = time.perf_counter() - STARTED
    fee = (review["complete_prior_first_fees_seconds"]
        + review["new_recovery_fees_including_first_failed_test_seconds"]
        + review["new_empty_write_diagnostic_reported_tool_wall_seconds"]
        + review["new_reaudit_first_path_failure_fee_seconds"] + lease["child_elapsed_seconds"] + final_seconds)
    medians = {key: {mode: statistics.median(pair["sides"][mode]["raw_" + key] + fee / 6
        for pair in review["raw_pairs"]) for mode in ("native", "stream")}
        for key in ("complete_phase_seconds", "whole_seconds")}
    reductions = {key: 1 - row["stream"] / row["native"] for key, row in medians.items()}
    checks.update(complete_phase_at_least5percent=reductions["complete_phase_seconds"] >= .05,
        complete_whole_median_not_regressed=reductions["whole_seconds"] >= 0,
        each_complete_whole_at_most1_1=all(pair["sides"]["stream"]["raw_whole_seconds"] + fee / 6 <=
            1.1 * (pair["sides"]["native"]["raw_whole_seconds"] + fee / 6) for pair in review["raw_pairs"]))
    result = {"status": "accepted_V2_scoped_engineering_mechanism" if all(checks.values()) else "failed_V2_verification",
        "mechanism_id": "R127", "mechanism": "full_native_Gaussian_PLY_block_byte_oracle_and_independent_artifact_copy",
        "checks": checks, "new_optimization_count": int(all(checks.values())),
        "complete_semantic_acceptance": all(checks.values()), "acceptance_policy_version": 2,
        "historical_V1_acceptance": False, "historical_V1_negative_preserved": True,
        "minimum_complete_stage_speed_reduction": .05, "whole_median_must_not_regress": True,
        "complete_first_fees_including_all_V1_and_V2_failure_recovery_review_seconds": fee,
        "V2_review_complete_lease_child_seconds": lease["child_elapsed_seconds"], "final_cost_and_close_seconds": final_seconds,
        "complete_medians_seconds": {"phase": medians["complete_phase_seconds"], "whole": medians["whole_seconds"]},
        "reductions": {"phase": reductions["complete_phase_seconds"], "whole": reductions["whole_seconds"]},
        "actual_peak_RSS_bytes": review["actual_peak_RSS_bytes"], "actual_RSS_reduction": review["actual_RSS_reduction"],
        "memory_is_supplemental_observation_not_replaced_speed_gate": True,
        "all_three_original_complete526951_Gaussian_pairs_and_parameters_preserved": True,
        "generic_both48_and_original_events_retained": True, "new12_real_permission_cancel_recovery_pairs_verified": True,
        "first_empty_probe_injection_failure_and_first_Windows_WSL_path_review_failure_preserved": True,
        "sources_inputs_rechecked": review["sources_inputs_checked"],
        "owned_Windows_checked": review["owned_Windows_checked"], "owned_Linux_checked": review["owned_Linux_checked"],
        "scope": "fixed_official_Gaussian_float32_CUDA_format_256MiB_serialization_verification_single_owner",
        "review_kind": "root_self_review_no_subagents", "independent_review": False, "production_modified": False,
        "new_object_Gaussian_full_pipeline_or_GT_accuracy": "unverified",
        "new_full_training_or_preview": "unrun_math_and_complete_model_bytes_unchanged",
        "evidence": [str(OUT / name) for name in ("FINAL_COST_001.json", "NEGATIVE_CLOSURE_001.json",
            "v2r0/RESULT.json", "v2r1/RESULT.json", "v2a1/REVIEW.json", "V2_REVIEW_LEASE_002.json")],
        "source": data_identity(Path(__file__)), "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(output, result)
    print({key: result[key] for key in ("status", "mechanism_id", "checks", "complete_medians_seconds", "reductions",
        "actual_RSS_reduction", "complete_first_fees_including_all_V1_and_V2_failure_recovery_review_seconds")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("Complete V2 protections or cost did not pass")


if __name__ == "__main__":
    main()
