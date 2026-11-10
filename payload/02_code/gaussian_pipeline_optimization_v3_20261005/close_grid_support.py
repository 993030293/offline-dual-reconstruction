"""Recompute all cost and output gates before independent grid support review."""

import json
from pathlib import Path
import statistics
import time

import psutil
import pycolmap
import numpy as np

from machine_json import exact_json_values
from profile_grid_support import ROOT, compare_outputs
from runtime import atomic_json, data_identity


OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005"
FOLDER = OUT/"grid_support_001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    checks, rows, identity_rows, cost_rows = {}, [], {}, []
    generic = read(FOLDER/"generic_001/RESULT.json")
    checks["all_current_220_full_function_oracles_passed"] = (
        generic["status"] == "all_complete_camera_grid_oracles_passed"
        and generic["checked"] == len(generic["checks"]) == 220
        and all(value is True for value in generic["checks"].values())
        and all(row["all_fields_types_order_and_float_bits_exact"] for row in generic["rows"]))
    for identity in generic["sources"]+generic["full_real_pycolmap_fixture_inputs"]:
        identity_rows[identity["path"]] = identity
    for campaign in ("campaign_001", "external_001"):
        folder = FOLDER/campaign
        result, contract = read(folder/"RESULT.json"), read(folder/"CONTRACT.json")
        medians = {label: {field: statistics.median(pair["rows"][label][field] for pair in result["pairs"])
                          for field in ("phase_seconds", "whole_seconds")} for label in ("native", "grid")}
        exact_json_values(medians, result["medians"])
        reduction = 1-medians["grid"]["phase_seconds"]/medians["native"]["phase_seconds"]
        stored_reduction = result["phase_reduction" if campaign == "campaign_001" else "complete_component_reduction"]
        exact_json_values(reduction, stored_reduction)
        gates = {"three_complete_pairs": len(result["pairs"]) == 3,
                 "complete_phase_reduction_at_least10pct": reduction >= .1,
                 "each_whole_at_most1_1": all(pair["rows"]["grid"]["whole_seconds"] <=
                     1.1*pair["rows"]["native"]["whole_seconds"] for pair in result["pairs"]),
                 "whole_median_nonregressed": medians["grid"]["whole_seconds"] <= medians["native"]["whole_seconds"]}
        checks[campaign+"_all_frozen_cost_gates_recomputed"] = all(gates.values()) and all(result["checks"].values())
        for pair in result["pairs"]:
            for label, row in pair["rows"].items():
                leaf = read(row["source_bound_leaf_result"])
                share = result["first_prepare_seconds"]/6
                comparison = pair["complete_comparison_seconds_charged_each_side"]
                expected_whole = leaf["complete_leaf_before_receipt_seconds"]+row["first_child_start_receipt_and_close_seconds"]+share+comparison
                expected_phase = (leaf["phase_seconds"] if campaign == "campaign_001" else
                                  leaf["complete_leaf_before_receipt_seconds"])+row["first_child_start_receipt_and_close_seconds"]+share+comparison
                if abs(expected_whole-row["whole_seconds"]) > 1e-12 or abs(expected_phase-row["phase_seconds"]) > 1e-12:
                    raise AssertionError("Cold startup, receipt, comparison or process closing cost was omitted")
                expected_calls = 3053 if campaign == "campaign_001" else 360
                if (row["child_returncode"] != 0 or len(leaf["trace"]["calls"]) != expected_calls
                        or not leaf["trace"]["original_binding_restored"]
                        or any(call["status"] != "passed_original_full_camera_support_function"
                               for call in leaf["trace"]["calls"])):
                    raise AssertionError("Actual full support membership or restoration changed")
                if campaign == "campaign_001" and not leaf["original_sparse_PLY_writer_used"]:
                    raise AssertionError("The original PLY route was replaced")
            left, right = [Path(pair["rows"][label]["folder"]) for label in ("native", "grid")]
            if campaign == "campaign_001":
                current = compare_outputs(left, right)
                exact_json_values(current, pair["verification"])
                if (current["complete_byte_pairs"], current["complete_JSON_pairs"], current["complete_compared_bytes"]) != (1046, 4, 746975609):
                    raise AssertionError("The complete actual local export denominator changed")
            else:
                first, second = left/"SUPPORT.json", right/"SUPPORT.json"
                exact_json_values(read(first), read(second))
                if first.read_bytes() != second.read_bytes() or first.stat().st_size != pair["complete_support_bytes"]:
                    raise AssertionError("An entire external scientific support artifact changed")
                if (read(first)["registered_images"], read(first)["points"], read(first)["observations"]) != (360, 133020, 848219):
                    raise AssertionError("The complete external denominator changed")
        checks[campaign+"_all_complete_output_comparisons_and_cost_envelopes_exact"] = True
        snapshots = sorted((folder/"source_snapshot").iterdir())
        source_rows = contract["sources_and_inputs"][:len(snapshots)]
        checks[campaign+"_all_timed_source_snapshots_exact"] = all(
            data_identity(snapshot)["sha256"] == identity["sha256"]
            for snapshot, identity in zip(snapshots, source_rows, strict=True))
        for identity in contract["sources_and_inputs"]:
            prior = identity_rows.get(identity["path"])
            if prior is not None and prior != identity:
                raise AssertionError("Generic, local and external sources disagree")
            identity_rows[identity["path"]] = identity
        cost_rows.append({"campaign": campaign, "medians": medians, "phase_reduction": reduction,
                          "recomputed_frozen_gates": gates})
    checks["all_complete_frozen_source_input_and_fixture_bytes_unchanged"] = all(
        data_identity(Path(identity["path"])) == identity for identity in identity_rows.values())
    checks["actual_fixed_library_versions_unchanged"] = pycolmap.__version__ == "4.1.1" and np.__version__ == "2.3.5"
    owners, nodes = {}, []
    for prefix in ("grid_support_generic_001", "grid_support_profile_001", "grid_support_external_001"):
        lease, node = read(OUT/f"{prefix}_lease.json"), read(OUT/f"{prefix}_process.json")
        if (lease["status"] != "child_exited" or lease["returncode"] != 0 or not lease.get("lease_released_utc")
                or node["status"] != "completed" or node["returncode"] != 0
                or node["elapsed_seconds"] > 1800 or node["peak_tree_RSS_bytes"] > 8*1024**3):
            raise AssertionError("A real measured node or lease did not complete within its bounds")
        owners[(lease["guard_pid"], lease["guard_creation_time"])] = "lease_guard"
        owners[(lease["child_pid"], lease["child_creation_time"])] = "lease_child"
        for item in node["owned_process_identities"]:
            owners[(item["pid"], item["creation_time"])] = "recorded_actual_child"
        nodes.append({"prefix": prefix, "status": node["status"], "elapsed_seconds": node["elapsed_seconds"],
                      "peak_tree_RSS_bytes": node["peak_tree_RSS_bytes"], "queue_wait_excluded_seconds": lease["wait_seconds"]})
    for (pid, creation), role in owners.items():
        try:
            process = psutil.Process(pid)
            live = process.create_time() == creation and process.is_running()
            status = "matching_owned_identity_live" if live else "PID_reused_by_different_process"
        except psutil.NoSuchProcess:
            live, status = False, "recorded_owned_identity_gone"
        rows.append({"pid": pid, "creation_time": creation, "role": role,
                     "matching_owned_identity_live": live, "status": status})
    checks["all_recorded_measured_process_identities_closed"] = not any(row["matching_owned_identity_live"] for row in rows)
    size = sum(path.stat().st_size for path in FOLDER.rglob("*") if path.is_file())
    checks["owned_output_budget_preserved"] = size < 30*1024**3
    result = {"status": "local_and_external_cost_output_source_and_resource_gates_passed_pending_independent_review"
        if all(checks.values()) else "closure_failed", "checks": checks, "candidate": "R112", "counted": False,
        "accepted_count_contribution": 0, "recomputed_costs": cost_rows, "actual_own_nodes": nodes,
        "actual_owned_process_identities": rows, "complete_immutable_identity_count": len(identity_rows),
        "own_output_bytes": size, "separate_closure_review_seconds": time.perf_counter()-started,
        "independent_review_completed": False, "full_unknown_object_training_quality_or_GT_claim": False,
        "pure_export_original_training_inputs_exact_no_new_training_required": True,
        "production_modified": False, "WSL_GPU_or_services_started_in_this_route": 0,
        "source": data_identity(Path(__file__)), "parent50_goal_complete": False}
    atomic_json(FOLDER/"CLOSURE.json", result)
    print(json.dumps({"status": result["status"], "output_bytes": size,
                      "recorded_identities": len(rows), "source_and_input_identities": len(identity_rows)}))
    if not all(checks.values()):
        raise AssertionError("The complete grid support closure failed")


if __name__ == "__main__":
    main()
