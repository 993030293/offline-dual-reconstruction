"""Bound complete JSON savings using actual unallocated and oracle costs."""

import time

STARTED = time.perf_counter()

import argparse
import ast
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from machine_json import exact_json_values


OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/machine_json_001"
PARENT = OLD.parent
OWN = PARENT / "mj_v4_001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    result = read(OLD / "campaign_001/RESULT.json")
    contract = read(OLD / "campaign_001/CONTRACT.json")
    checks = {}
    try:
        original_identities = contract["sources_and_inputs"]
        checks["all_current_original_source_and_full_input_identities_exact"] = [data_identity(Path(row["path"])) for row in original_identities] == original_identities
        source_dir = output / "executed_sources"
        source_dir.mkdir()
        for row in original_identities[:4]:
            path = Path(row["path"])
            if (OLD / "campaign_001/source_snapshot" / path.name).read_bytes() != path.read_bytes():
                raise AssertionError("An original actual source snapshot differs")
            shutil.copyfile(path, source_dir / path.name)
            ast.parse(path.read_text(encoding="utf-8-sig"))
        shutil.copyfile(Path(__file__), source_dir / Path(__file__).name)
        checks["same_three_alternating_real_complete_export_pairs_reused"] = len(result["pairs"]) == 3 and [list(pair["rows"]) for pair in result["pairs"]] == [["native", "compact"], ["compact", "native"], ["native", "compact"]]
        counts = []
        for pair in result["pairs"]:
            folder = OLD / "campaign_001" / ("trial" + str(pair["trial"]))
            for name in ("camera_path.json", "RECEIPT.json"):
                values = [read(folder / label / name) for label in ("native", "compact")]
                exact_json_values(*values)
                if name == "camera_path.json":
                    if type(values[0]) is not list:
                        raise AssertionError("The original camera path is not its documented list schema")
                    counts.append(len(values[0]))
            for label, row in pair["rows"].items():
                trace = row["trace"]
                phase = trace["setup_seconds"] + trace["restore_seconds"] + sum(call["write_seconds"] + call["complete_validation_seconds"] for call in trace["calls"])
                phase += pair["paired_check_seconds_charged_each_side"] + result["first_preparation_seconds"] / 6
                exact_json_values(phase, row["phase_seconds"])
                if not trace["original_binding_restored"] or len(trace["calls"]) != 2:
                    raise AssertionError("The original writer binding or full output count differs")
        checks["all_three_full1016_paths_types_bits_order_and_complete_receipts_exact"] = counts == [1016, 1016, 1016]
        for label in ("native", "compact"):
            for name in ("phase_seconds", "whole_seconds"):
                exact_json_values(statistics.median(pair["rows"][label][name] for pair in result["pairs"]), result["medians"][label][name])
        checks["all_original_side_and_median_phase_arithmetic_bit_exact"] = True
        generic = read(OLD / "oracle_001/RESULT.json")
        checks["old_generic28_and_strict_whole_failure_preserved"] = generic["status"] == "all_real_JSON_data_publication_oracles_passed" and result["status"] == "complete_native_export_cost_gate_failed" and not result["checks"]["complete_whole_median_nonregressed"]
        formal = read(PARENT / "machine_json_001_profile_lease.json")
        oracle = read(PARENT / "machine_json_001_oracle_lease.json")
        checks["both_actual_old_leases_successfully_exited_and_released"] = all(row["status"] == "child_exited" and row["returncode"] == 0 and row.get("lease_released_utc") for row in (formal, oracle))
        # Each pair comparison was billed to both sides but ran only once.
        already_allocated_actual = sum(row["whole_seconds"] for pair in result["pairs"] for row in pair["rows"].values()) - sum(pair["paired_check_seconds_charged_each_side"] for pair in result["pairs"])
        residue = formal["child_elapsed_seconds"] - already_allocated_actual
        if residue < 0:
            raise AssertionError("The formal components exceed the actual closed parent lease")
        original_fee = residue + oracle["child_elapsed_seconds"]
        elapsed = time.perf_counter() - STARTED
        fee = original_fee + elapsed
        phase = {label: result["medians"][label]["phase_seconds"] + fee / 6 for label in ("native", "compact")}
        whole = {label: result["medians"][label]["whole_seconds"] + fee / 6 for label in ("native", "compact")}
        reduction = 1 - phase["compact"] / phase["native"]
        if not all(checks.values()):
            raise AssertionError("A saved source, semantic, cost or resource check failed")
        record = {"status": "saved_complete_cost_audit_passed_actual_exit_and_final_bound_pending",
                  "checks": checks, "original_formal_already_allocated_actual_seconds": already_allocated_actual,
                  "actual_formal_unallocated_seconds": residue, "old_oracle_seconds": oracle["child_elapsed_seconds"],
                  "original_additional_fee_lower_bound_seconds": original_fee,
                  "audit_internal_seconds": elapsed, "preliminary_additional_known_fee_seconds": fee,
                  "preliminary_stage_reduction_upper_bound": reduction,
                  "preliminary_phase_medians_seconds": phase, "preliminary_whole_medians_seconds": whole,
                  "first_preparation_and_pair_compare_not_double_charged": True,
                  "formal_three_real_pairs_reused_not_new": True,
                  "browser_recovery_and_old_unmetered_review_costs_not_yet_in_lower_bound": True,
                  "new_optimization_count": 0, "source_input_identities": original_identities,
                  "audit_source": data_identity(Path(__file__))}
        atomic_json(output / "RESULT.json", record)
        print(json.dumps({"status": record["status"], "checks": len(checks),
                          "known_fee_lower_bound": fee, "stage_upper_bound": reduction, "formal_residue": residue}))
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"checks": checks, "error_type": type(error).__name__,
                                               "error": str(error), "actual_audit_seconds_before_failure": time.perf_counter() - STARTED,
                                               "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
