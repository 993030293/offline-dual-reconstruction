"""Audit the preserved first failure, identities and owned process drain."""

import argparse
import ast
import gc
import json
from pathlib import Path
import time

STARTED = time.perf_counter()
import psutil
from profile_direct_schur import ROOT, source_paths, input_paths
from runtime import atomic_json, data_identity
from run_direct_schur_campaign import drained


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result, sides = read(args.campaign / "CAMPAIGN_RESULT.json"), read(args.campaign / "CAMPAIGN_SIDES.json")
    checks = {"first_failure_pair1_whole": result["failure"]["node"] == "pair1" and result["failure"]["phase"] == "each_whole_cost_guard",
              "registered_limit_unchanged": result["failure"]["limit"] == 1.1 and result["failure"]["observed_ratio"] > 1.1,
              "exactly_four_metric_nodes": len(sides) == 4 and result["completed_node_count"] == 4,
              "third_pair_absent": not any((args.campaign / name).exists() for name in ("pair2_auto", "pair2_direct")),
              "count_zero": result["counted"] is False,
              "all_contingent_experiments_not_run": all(value == "not_run_after_failure" for name, value in result["dependent_status"].items() if name != "root_independent_review"),
              "four_actual_generic_solves": read(args.campaign / "backend_c001/RESULT.json")["actual_solve_count"] == 4,
              "repaired_oracle_complete": read(args.campaign / "oracle_c002/RESULT.json")["status"] == "full_readback_and_projection_oracle_passed",
              "first_checker_error_preserved": (args.campaign / "oracle_c001/FAILURE.json").exists() and
                (args.campaign / "oracle_c001/profile_source_before_repair.py").exists() and
                (args.campaign / "oracle_c001/checker_source_before_repair.py").exists()}
    live_sources, live_inputs = [data_identity(path) for path in source_paths()], [data_identity(path) for path in input_paths()]
    cases = []
    for side in sides:
        name = side["node"]
        contract = read(args.campaign / name / "CONTRACT.json")
        checks[name+"_all_current_sources_inputs_match"] = contract["source"] == live_sources and contract["inputs"] == live_inputs
        checks[name+"_guard_budget_and_drain"] = (side["resource"]["status"] == "completed" and
            side["resource"]["maximum_seconds"] == 1800 and side["resource"]["maximum_RSS_bytes"] == 8*1024**3
            and side["resource"]["elapsed_seconds"] <= 1800 and side["resource"]["peak_tree_RSS_bytes"] <= 8*1024**3
            and drained(side["resource"], side["lease"]))
        checks[name+"_all_original_final_guards"] = all(read(args.campaign / name / "GUARDS.json").values())
        checks[name+"_restore_and_source_recheck"] = side["result"]["interface_restored"] and side["result"]["sources_inputs_unchanged"]
        for label in ("visual_metric_aligned", "weak_position_prior"):
            checks[name+"_"+label+"_full_readback"] = all(read(args.campaign / name / (label+"_FULL_READBACK.json"))["checks"].values())
            item = side["result"]["original_result"]["results"][label]
            cases.append({"node": name, "case": label, "model": item["model"],
                          "role": "unaccepted_R114_performance_rejected_no_export_or_training",
                          "original_metric_interface_guard_passed": True})
    for name in ("backend_c001", "oracle_c001", "oracle_c002"):
        checks[name+"_drained"] = drained(read(args.campaign/(name+"_RESOURCE.json")), read(args.campaign/(name+"_LEASE.json")))
    syntax = []
    for name in ("direct_schur.py", "check_direct_schur.py", "check_direct_schur_oracle.py", "profile_direct_schur.py", "run_direct_schur_campaign.py", "close_direct_schur.py"):
        path = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005" / name
        ast.parse(path.read_text(encoding="utf-8")); syntax.append(str(path))
    atomic_json(args.campaign / "MAP_QUARANTINE.json", {"counted": False, "first_failure": result["failure"], "cases": cases,
                "root_negative_result_review": "pending_root_review", "promotion": "rejected_cost_guard"})
    gc.collect()
    atomic_json(args.output / "RESULT.json", {"status": "negative_cost_result_audit_passed" if all(checks.values()) else "closure_audit_failed",
                "checks": checks, "syntax_checked": syntax, "counted": False, "source": data_identity(Path(__file__)),
                "whole_seconds": time.perf_counter()-STARTED, "root_review_is_pending_not_a_skipped_experiment": True,
                "this_closure_cost_reported_separately_after_performance_failure": True})
    if not all(checks.values()): raise AssertionError("Negative-result resource or identity closure audit failed")


if __name__ == "__main__": main()
