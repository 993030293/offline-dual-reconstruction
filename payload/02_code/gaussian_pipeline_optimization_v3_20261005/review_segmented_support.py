"""Review current complete exports, historical failures and actual resource ownership."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import statistics

import psutil

from profile_sparse_ply_oracle import ROOT, compare_outputs
from machine_json import exact_json_values
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    current_identities, historical_identities = {}, {}
    core = Path(__file__).parent / "segmented_camera_support.py"
    archived = [data_identity(OUT / folder / core.name)
                for folder in ("source_history_001", "source_history_002")]

    def require(row, allow_historical=False):
        path = Path(row["path"])
        key = str(path.resolve()).lower()
        actual = current_identities.get(key)
        if actual is None:
            actual = data_identity(path)
            current_identities[key] = actual
        if (actual["bytes"], actual["sha256"]) == (row["bytes"], row["sha256"]):
            return
        if allow_historical and path.resolve() == core.resolve():
            if any((item["bytes"], item["sha256"]) == (row["bytes"], row["sha256"])
                   for item in archived):
                historical_identities[row["sha256"]] = row
                return
        raise AssertionError("A current or retained source/input identity changed: " + str(path))

    def verify_record(record, historical=False):
        for row in record.get("sources", []) + record.get("inputs", []):
            require(row, allow_historical=historical)

    def verify_source_copies(path, record):
        for ordinal, row in enumerate(record["sources"]):
            copied = path.parent / "source_snapshot" / (f"{ordinal:03d}_" + Path(row["path"]).name)
            actual = data_identity(copied)
            if (actual["bytes"], actual["sha256"]) != (row["bytes"], row["sha256"]):
                raise AssertionError("A complete source snapshot changed: " + str(copied))
        return len(record["sources"])

    generic = {name: read(OUT / name / "RESULT.json") for name in ("g6", "g7", "g5")}
    for record in generic.values():
        verify_record(record)
    for name in ("g0", "g1", "g2", "g3", "g4"):
        verify_record(read(OUT / name / "RESULT.json"), historical=True)
    first_failure = read(OUT / "c0/FAILURE.json")
    failed_leaf = read(OUT / "c0/t0/segmented/FAILURE.json")
    reproduction = read(OUT / "g4/RESULT.json")
    verify_record(first_failure, historical=True)
    verify_record(failed_leaf, historical=True)
    campaign = read(OUT / "c1/RESULT.json")
    verify_record(campaign)
    external = read(OUT / "x0/RESULT.json")
    current_nodes, saved_source_copies, comparisons = [], 0, []
    for pair in campaign["pairs"]:
        pair_folder = OUT / "c1" / f"t{pair['pair']}"
        nodes = {}
        for mode in ("native", "segmented"):
            path = pair_folder / mode / "RESULT.json"
            record = read(path)
            verify_record(record)
            saved_source_copies += verify_source_copies(path, record)
            current_nodes.append(record)
            nodes[mode] = record
            restoration = read(path.parent / "BINDING_RESTORATION.json")
            if not all(value for key, value in restoration.items() if key != "new_optimization_count"):
                raise AssertionError("A complete export did not restore its private bindings")
        exact_json_values(nodes["native"]["sources"], nodes["segmented"]["sources"])
        exact_json_values(nodes["native"]["inputs"], nodes["segmented"]["inputs"])
        comparison = compare_outputs(pair_folder / "native/export", pair_folder / "segmented/export")
        if comparison != pair["comparison"]:
            raise AssertionError("A current complete output comparison changed")
        comparisons.append(comparison)
    external_nodes = []
    for pair in external["pairs"]:
        nodes = {}
        for mode in ("native", "segmented"):
            path = OUT / "x0" / f"t{pair['pair']}" / mode / "RESULT.json"
            record = read(path)
            verify_record(record)
            saved_source_copies += verify_source_copies(path, record)
            external_nodes.append(record)
            nodes[mode] = record
        for key in ("rows", "sources", "inputs", "before", "after"):
            exact_json_values(nodes["native"][key], nodes["segmented"][key])
        if nodes["native"]["before"] != nodes["native"]["after"]:
            raise AssertionError("The external native owner state changed")
    for relative in ("p0/native/RESULT.json", "p0/segmented/RESULT.json",
                     "c0/t0/native/RESULT.json", "c0/t0/segmented/FAILURE.json"):
        path = OUT / relative
        record = read(path)
        verify_record(record, historical=True)
        saved_source_copies += verify_source_copies(path, record)
    original_core = lambda record: next(row for row in record["sources"] if Path(row["path"]).name == core.name)
    initial_core = original_core(read(OUT / "g0/RESULT.json"))
    failing_core = original_core(failed_leaf)
    archived_sources_verified = all((actual["bytes"], actual["sha256"]) == (row["bytes"], row["sha256"])
                                    for actual, row in zip(archived, (initial_core, failing_core)))
    archived_sources_verified = archived_sources_verified and original_core(reproduction)["sha256"] == failing_core["sha256"]
    windows, costs, prior_leases = {}, [], []
    for path in sorted(OUT.glob("*LEASE*.json")):
        if path.name.startswith("REVIEW_LEASE_"):
            continue
        row = read(path)
        expected_code = 1 if path.name == "CAMPAIGN_LEASE_001.json" else 0
        if row["status"] != "child_exited" or row["returncode"] != expected_code or not row.get("lease_released_utc"):
            raise AssertionError("A measured prior lease did not close its actual child")
        prior_leases.append(str(path))
        costs.append({"path": str(path), "seconds": row["child_elapsed_seconds"],
                      "returncode": row["returncode"], "scope": "root_lease_child_no_duplicate_nested_guards"})
        for prefix in ("guard", "child"):
            identity = {"pid": row[prefix + "_pid"], "creation_time": row[prefix + "_creation_time"]}
            windows[(identity["pid"], identity["creation_time"])] = identity
    guard_paths = [path for path in OUT.glob("*GUARD*.json") if not path.name.startswith("REVIEW_GUARD_")]
    guard_paths += [OUT / branch / f"t{ordinal}" / f"{mode}_guard.json"
                    for branch, count in (("c0", 1), ("c1", 3), ("x0", 3))
                    for ordinal in range(count) for mode in ("native", "segmented")]
    for path in guard_paths:
        resource = read(path)
        expected_code = 1 if path.name == "CAMPAIGN_GUARD_001.json" or path == OUT / "c0/t0/segmented_guard.json" else 0
        if resource["returncode"] != expected_code or resource["elapsed_seconds"] > 1750:
            raise AssertionError("A measured bounded child had an unexpected exit or duration")
        if resource["peak_tree_RSS_bytes"] > 8 * 1024**3:
            raise AssertionError("A measured child exceeded the unchanged resource budget")
        require(resource["source"])
        for identity in resource["owned_process_identities"]:
            windows[(identity["pid"], identity["creation_time"])] = identity
    live = []
    for identity in windows.values():
        try:
            process = psutil.Process(identity["pid"])
            if process.create_time() == identity["creation_time"] and process.is_running():
                live.append(identity)
        except psutil.NoSuchProcess:
            pass
    node_total = sum(pair["nodes"][mode]["complete_whole_seconds"]
                     for pair in campaign["pairs"] for mode in ("native", "segmented"))
    current_campaign_cost = next(row["seconds"] for row in costs if Path(row["path"]).name == "CAMPAIGN_LEASE_002.json")
    campaign_overhead = current_campaign_cost - node_total
    first_fees = sum(row["seconds"] for row in costs if Path(row["path"]).name != "CAMPAIGN_LEASE_002.json")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    output_bytes = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    segmented_nodes = [row for row in current_nodes if row["mode"] == "segmented"]
    checks = {
        "current_generic16_and_private_binding7_complete": len(generic["g6"]["checks"]) == 16
            and len(generic["g7"]["checks"]) == 7 and all(all(row["checks"].values()) for row in generic.values()),
        "current_actual_lifecycle24_and_same_failure_recovery_complete": len(generic["g5"]["checks"]) == 24
            and generic["g5"]["checks"]["same_actual_deregistered_failure_repaired_complete_values"]
            and generic["g5"]["checks"]["fresh_directory_recovery_complete_original_values"],
        "initial_failure_and_both_historical_sources_retained": first_failure["state"]["complete_sides"] == 1
            and first_failure["state"]["completed_pairs"] == 0 and archived_sources_verified
            and failed_leaf["error"] == "ValueError('The native snapshot image identities changed')",
        "real_memory9_registered8_snapshot8_failure_reproduced": all(reproduction["checks"].values())
            and (reproduction["in_memory_images"], reproduction["registered_images"], reproduction["native_snapshot_images"]) == (9, 8, 8),
        "three_fresh_current_complete_exports_all1050_outputs_exact": len(comparisons) == 3
            and all(row["complete_byte_pairs"] == 1046 and row["complete_JSON_pairs"] == 4 for row in comparisons),
        "all_six_current_full_export_checks_passed": len(current_nodes) == 6
            and all(all(row["checks"].values()) for row in current_nodes),
        "all3053_supports_three_actual_distinct_states": all([stage["stage"] for stage in row["trace"]["stages"]] == ["before", "after", "final"]
            and [stage["images"] for stage in row["trace"]["stages"]] == [1021, 1016, 1016]
            and row["trace"]["no_cross_filter_or_undistort_reuse"] for row in segmented_nodes),
        "all_registered_and_omitted_state_guards_verified": all(stage["details"]["route"] == "whole_map_segmented_support"
            and stage["details"]["closed"] and stage["details"]["index"]["complete_native_state_unchanged"]
            and stage["details"]["index"]["omitted_image_metadata_pixels_and_bindings_exact"]
            for row in segmented_nodes for stage in row["trace"]["stages"]),
        "actual_after_state_five_unregistered_images_verified": all([stage["details"]["index"]["unregistered_images"]
            for stage in row["trace"]["stages"]] == [0, 5, 0] for row in segmented_nodes),
        "all_current_source_and_input_identities_verified": bool(current_identities),
        "all184_complete_current_and_historical_source_copies_verified": saved_source_copies == 184,
        "fixed_code_three_full360_external_component_pairs_exact": len(external["pairs"]) == 3
            and len(external_nodes) == 6 and all(all(row["checks"].values()) and len(row["rows"]) == 360 for row in external_nodes),
        "external_all_candidates_used_actual_segmented_algorithm": all(row["trace"]["route"] == "whole_map_segmented_support"
            for row in external_nodes if row["mode"] == "segmented"),
        "no_external_GT_or_complete_new_object_claim": not external["GT_metric_or_complete_new_object_quality_claim"]
            and all(not row["independent_GT_metric_pose_or_new_object_training_claim"] for row in external_nodes),
        "actual_prior_Windows_instances_gone": not live,
        "all_fourteen_prior_leases_released": len(prior_leases) == 14,
        "all_first_failure_external_and_campaign_fees_recorded_without_duplicates": first_fees > 0
            and campaign_overhead >= 0 and len(costs) == 14,
        "whole_raw_median_and_all_original_pair_guards_retained": statistics.median(pair["nodes"]["segmented"]["complete_whole_seconds"] for pair in campaign["pairs"])
            <= statistics.median(pair["nodes"]["native"]["complete_whole_seconds"] for pair in campaign["pairs"])
            and all(pair["whole_ratio"] <= 1.1 for pair in campaign["pairs"]),
        "unchanged_group_output_and_free_space_limits": output_bytes <= 30 * 1024**3
            and shutil.disk_usage(OUT).free >= 20 * 1024**3,
        "parent17_unique_and_v2_5percent_unchanged_before_promotion": state["additional_accepted_count"] == 17
            and len(set(state["accepted_mechanisms"])) == 17 and state["minimum_complete_stage_speed_reduction_current"] == .05,
        "new_full_training_not_run_for_byte_identical_training_inputs": not any((OUT / name).exists() for name in ("training", "t0", "t1", "t2")),
    }
    result = {
        "status": "root_review_passed_final_complete_cost_and_own_close_pending" if all(checks.values()) else "failed_root_review",
        "review_kind": "root_self_review_no_subagents_or_side_chats",
        "checks": checks, "unique_current_source_input_identities": len(current_identities),
        "retained_historical_core_versions": len(historical_identities), "saved_source_copies": saved_source_copies,
        "complete_output_comparisons": comparisons, "prior_leases": prior_leases,
        "Windows_process_identities_checked": list(windows.values()), "live_Windows": live,
        "Linux_execution_for_this_candidate": "none_no_WSL_GPU_node_claim",
        "root_costs_without_nested_double_counting": costs, "current_complete_campaign_lease_seconds": current_campaign_cost,
        "current_six_node_seconds": node_total, "current_campaign_setup_compare_publish_and_close_seconds": campaign_overhead,
        "complete_prior_first_failure_diagnostic_lifecycle_and_external_fees_seconds": first_fees,
        "output_bytes": output_bytes, "free_bytes": shutil.disk_usage(OUT).free,
        "own_review_cost_and_process_closure": "pending_actual_wrapper_exit_and_lease_release",
        "claim_boundary": "three_complete_local_exports_and_fixed_existing_external360_component_no_independent_metric_GT_or_new_object_training",
        "runtime_cost_boundary": "all_actual_measured_algorithm_and_required_validation_nodes_inclusive_of_first_failures; manual_source_authoring_knowledge_lookup_and_document_writing_are_not_algorithm_runtime",
        "review_body_seconds_before_final_publication": time.perf_counter() - STARTED,
        "new_optimization_count": 0,
    }
    atomic_json(args.output / "REVIEW.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "unique_current_source_input_identities", "saved_source_copies", "new_optimization_count")}))
    if not all(checks.values()):
        raise AssertionError("The current whole-map segmented support review failed its evidence scope")


if __name__ == "__main__":
    main()
