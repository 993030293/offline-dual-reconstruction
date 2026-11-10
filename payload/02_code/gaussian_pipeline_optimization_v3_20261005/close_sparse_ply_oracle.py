"""Close a complete negative sparse-export campaign using actual receipts."""

import json
from pathlib import Path
import statistics
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005"
FOLDER = OUT/"sparse_ply_oracle_001"
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    result = read(FOLDER/"campaign_002/RESULT.json")
    if result["status"] != "complete_export_cost_guard_failed" or len(result["pairs"]) != 3:
        raise ValueError("Negative closure requires the actual full three-pair cost rejection")
    medians = {label: {field: statistics.median(pair["rows"][label][field] for pair in result["pairs"])
                      for field in ("phase_seconds", "whole_seconds")} for label in ("native", "columns")}
    reduction = 1-medians["columns"]["phase_seconds"]/medians["native"]["phase_seconds"]
    gate = {"phase_reduction_at_least10pct": reduction >= .1,
            "each_whole_at_most1_1": all(pair["rows"]["columns"]["whole_seconds"] <=
                1.1*pair["rows"]["native"]["whole_seconds"] for pair in result["pairs"]),
            "whole_median_nonregressed": medians["columns"]["whole_seconds"] <= medians["native"]["whole_seconds"],
            "all_original_bindings_and_all_output_contents_exact": all(
                pair["verification"]["complete_byte_pairs"] == 1046
                and pair["verification"]["complete_JSON_pairs"] == 4
                and pair["verification"]["complete_compared_bytes"] == 746975609
                and all(row["trace"]["original_binding_restored"] and row["child_returncode"] == 0
                        and len(row["trace"]["calls"]) == 1
                        and row["trace"]["calls"][0]["points"] == 75767
                        and row["trace"]["calls"][0]["status"] == "passed_complete_official_roundtrip"
                        for row in pair["rows"].values()) for pair in result["pairs"])}
    checks = {"stored_medians_and_all_gate_booleans_recomputed_exact": medians == result["medians"]
                    and gate == result["checks"] and reduction == result["phase_reduction"],
              "original_phase_speed_gate_failed_not_relaxed": not gate["phase_reduction_at_least10pct"],
              "all_actual_export_contents_and_original_bindings_exact": gate["all_original_bindings_and_all_output_contents_exact"]}
    oracle_paths = [FOLDER/"generic_002/RESULT.json", FOLDER/"rejection_001/RESULT.json",
                    FOLDER/"compare_guard_001/RESULT.json"]
    checks["all_current_27_plus3_plus13_oracles_passed"] = all(
        all(read(path)["checks"].values()) for path in oracle_paths)
    contract = read(FOLDER/"campaign_002/CONTRACT.json")
    names = sorted((FOLDER/"campaign_002/source_snapshot").iterdir())
    source_rows = contract["sources_and_inputs"][:len(names)]
    checks["all_actual_timed_source_snapshots_and_current_code_unchanged"] = all(
        data_identity(Path(identity["path"]))["sha256"] == identity["sha256"]
        and data_identity(snapshot)["sha256"] == identity["sha256"]
        for snapshot, identity in zip(names, source_rows, strict=True))
    prefixes = ("sparse_ply_oracle_001_generic", "sparse_ply_oracle_002_generic",
                "sparse_ply_rejection_001", "sparse_ply_profile_001",
                "sparse_export_compare_001", "sparse_ply_profile_002")
    owners, nodes = {}, []
    for prefix in prefixes:
        lease, node = read(OUT/f"{prefix}_lease.json"), read(OUT/f"{prefix}_process.json")
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
            raise AssertionError("An actual own lease has not closed")
        owners[(lease["guard_pid"], lease["guard_creation_time"])] = "lease_guard"
        owners[(lease["child_pid"], lease["child_creation_time"])] = "lease_child"
        for item in node["owned_process_identities"]:
            owners[(item["pid"], item["creation_time"])] = "observed_actual_owned_child"
        nodes.append({"prefix": prefix, "status": node["status"], "returncode": node["returncode"],
                      "elapsed_seconds": node["elapsed_seconds"], "peak_tree_RSS_bytes": node["peak_tree_RSS_bytes"],
                      "lease_wait_not_algorithm_seconds": lease["wait_seconds"]})
    process_rows = []
    for (pid, created), role in owners.items():
        try:
            process = psutil.Process(pid)
            matching_live = process.create_time() == created and process.is_running()
            status = "matching_owned_process_still_live" if matching_live else "PID_reused_by_different_process"
        except psutil.NoSuchProcess:
            matching_live, status = False, "recorded_owned_identity_gone"
        process_rows.append({"pid": pid, "creation_time": created, "role": role,
                             "matching_owned_identity_live": matching_live, "status": status})
    checks["all_recorded_own_PID_and_creation_time_identities_closed"] = not any(
        row["matching_owned_identity_live"] for row in process_rows)
    checks["formal_node_bounded_and_completed"] = (nodes[-1]["status"] == "completed"
        and nodes[-1]["returncode"] == 0 and nodes[-1]["elapsed_seconds"] < 1800
        and nodes[-1]["peak_tree_RSS_bytes"] <= 8*1024**3)
    output_bytes = sum(path.stat().st_size for path in FOLDER.rglob("*") if path.is_file())
    closure = {"status": "negative_three_pair_phase_speed_closed" if all(checks.values()) else "closure_failed",
               "candidate": "R111", "accepted_count_contribution": 0, "counted": False,
               "checks": checks, "recomputed_cost_gates": gate, "recomputed_medians": medians,
               "complete_phase_reduction": reduction, "actual_own_nodes": nodes,
               "actual_owned_process_identities": process_rows, "output_bytes": output_bytes,
               "scope": "complete_original_export_fresh_child_entry_and_all_costed_sparse_PLY_protections",
               "campaign001_validator_API_failure_preserved_not_performance_evidence": True,
               "remaining_conditional_work": {"Oxford_component": "not_run_first_required_local_phase_cost_gate_failed",
                    "new_training": "not_run_pure_export_outputs_exact_and_candidate_rejected",
                    "browser_preview": "not_run_candidate_rejected_no_preview_artifacts_changed",
                    "independent_review": "not_run_this_is_separate_root_self_review"},
               "WSL_or_CUDA_tasks_started_in_this_route": 0, "services_started_in_this_route": 0,
               "production_modified": False, "parent50_goal_complete": False,
               "source": data_identity(Path(__file__))}
    atomic_json(FOLDER/"CLOSURE.json", closure)
    print(json.dumps({"status": closure["status"], "phase_reduction": reduction,
                      "owned_identities": len(process_rows), "output_bytes": output_bytes}))
    if not all(checks.values()):
        raise AssertionError("Actual negative sparse export closure is incomplete")


if __name__ == "__main__":
    main()
