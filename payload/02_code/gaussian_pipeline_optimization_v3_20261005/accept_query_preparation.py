"""Accept bounded reference preparation only after full ordered localization."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "oracle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign, oracle = read(args.campaign/"RESULT.json"), read(args.oracle/"RESULT.json")
    names = {"all10_empty_single_varied_sequences_exact_once_and_single_producer",
        "producer_error_at_original_consumption_position_and_no_following_preparation",
        "consumer_failure_preserved", "consumer_fault_cancels_waits_and_closes_before_outer_caches",
        "early_close_cancels_waits_and_closes_before_outer_caches",
        "incumbent_native_source_composition_and_error_restoration",
        "serial_native_source_composition_and_error_restoration",
        "parallel_native_source_composition_and_error_restoration", "all_owned_producer_threads_closed"}
    lifecycles = []
    for trial in range(3):
        for mode in ("incumbent", "serial", "parallel"):
            item = read(args.campaign/f"trial{trial}"/mode/"PREPARATION.json")
            trace = item["trace"]
            lifecycles.append(item["producer_closed_inside_native_main_before_outer_reference_cache_exit"]
                and trace["query_loop_seconds"] > 0 and trace["composition_seconds"] > 0
                and item["complete_original_index_trace"]["index"]["complete_native_state_unchanged"]
                and all(source == data_identity(Path(source["path"])) for source in item["sources"])
                and (mode == "incumbent" or trace["producer"]["closed"]))
    checks = {"all_three_actual_complete_triplets_and_original24_48_guards":
            campaign["status"] == "all_actual_prepared_query_guards_passed_acceptance_pending"
            and len(campaign["triplets"]) == 3 and all(campaign["checks"].values())
            and all(all(row["checks"].values()) for row in campaign["triplets"]),
        "complete_query_stage_reduces10percent_with_cold_index_and_final_native_verification":
            campaign["reductions"]["complete_query_stage_seconds"] >= .1
            and campaign["cold_index_and_final_native_map_verification_and_all_preparation_lifecycle_in_stage"],
        "each_complete_entry_protected_and_whole_median_cannot_regress":
            campaign["reductions"]["whole_seconds"] >= 0
            and all(row["whole_seconds"]["parallel"] <= 1.1*row["whole_seconds"]["incumbent"]
                for row in campaign["triplets"]),
        "all9_named_sequence_ownership_error_shutdown_and_restoration_oracles":
            oracle["status"] == "passed" and set(oracle["checks"]) == names and all(oracle["checks"].values()),
        "actual_route_bound_to_the_complete_original_source_oracle":
            campaign["sources"][1] == oracle["sources"][1] and campaign["sources"][5] == oracle["sources"][2],
        "all_current_sources_and_all_nine_complete_lifecycles_preserved": all(lifecycles)
            and all(source == data_identity(Path(source["path"])) for source in [*campaign["sources"], *oracle["sources"]]),
        "single_producer_and_caller_geometry_preserved_with_failed_F_parallel_disabled":
            campaign["reference_preparation_single_producer_SIFT_and_PnP_on_caller"]
            and campaign["same_current_R90_R92_R94_original24_48_and_failed_R101_disabled"],
        "offline_only_no_object_shape_GT_or_new_selection_information":
            campaign["original_offline_queries_no_GT_or_new_selection_information"]}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_verification",
        "mechanism": "R102_ORDERED_REFERENCE_PREPARATION_LOCALIZATION_PIPELINE",
        "additional_accepted_count": int(all(checks.values())), "checks": checks,
        "median": campaign["median"], "reductions": campaign["reductions"],
        "actual_next_preparations_overlapping_current_query": [row["actually_overlapping_next_preparations"] for row in campaign["triplets"]],
        "scope": "offline_owned_single_reference_producer_bounded2_original_ordered_SIFT_map_matching_PnP_commits",
        "actual_current_map_all24_development_queries48_excluded_inputs_preserved": True,
        "generic_contract_checked_new_object_whole_pipeline_pending": True,
        "no_new_training_or_geometry_accuracy_claim": True,
        "whole_seconds_polling_resolution_about0_5sec": True,
        "independent_review": False, "campaign": str(args.campaign), "oracle": str(args.oracle),
        "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({key: result[key] for key in ("status", "checks", "additional_accepted_count", "median")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original complete reference preparation acceptance guard failed")


if __name__ == "__main__":
    main()
