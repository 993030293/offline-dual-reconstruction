"""Reaudit preserved complete monitor pairs under the preregistered V4 tradeoff."""

import argparse
import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

import psutil


STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nm001"
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nm_v4_001"
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from nvml_owned_node import linux_identity_check, remaining
from nvml_memory_monitor_002 import NativeMemory
from run_nvml_monitor_campaign_002 import compare_pair


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_fault_only_main(path):
    spec = importlib.util.spec_from_file_location("review_original_owned_node", CODE / "nvml_owned_node.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = ast.parse(inspect.getsource(module.main))
    candidate = ast.parse(path.read_text(encoding="utf-8"))
    before = [item for item in ast.walk(original) if isinstance(item, ast.ExceptHandler)
              and isinstance(item.type, ast.Name) and item.type.id == "BaseException"]
    after = [item for item in ast.walk(candidate) if isinstance(item, ast.ExceptHandler)
             and isinstance(item.type, ast.Name) and item.type.id == "BaseException"]
    if len(before) != 1 or len(after) != 1:
        raise AssertionError("The private controller exception contract changed")
    if ast.dump(ast.Module(body=[after[0].body[0], *after[0].body[4:]], type_ignores=[])) != ast.dump(
            ast.Module(body=before[0].body, type_ignores=[])):
        raise AssertionError("The original owned drain body changed beyond the new census")
    after[0].body = copy.deepcopy(before[0].body)
    if ast.dump(original) != ast.dump(candidate):
        raise AssertionError("A normal sampling, budget, launch or close path changed")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "source_snapshot.py")
    old = read(OLD / "c1/RESULT.json")
    old_close = read(OLD / "NEGATIVE_CLOSURE_001.json")
    generic = read(OWN / "g0/RESULT.json")
    checks = {"old_strict_whole_failure_preserved": old["status"] == "failed_verification"
              and old_close["new_optimization_count"] == 0 and not old["checks"]["complete_whole_median_not_regressed"],
              "old21_negative_scope_checks_preserved": len(old_close["checks"]) == 21 and all(old_close["checks"].values()),
              "current_original_four_real_GPU_states_and7_guards_exact": generic["status"] == "passed"
              and len(generic["rows"]) == 4 and len(generic["checks"]) == 7 and all(generic["checks"].values()),
              "current_driver_library_single_device_ABI_unchanged": generic["metadata"] == read(OLD / "g1/RESULT.json")["metadata"],
              "same_three_actual_ordered_complete_pairs": len(old["pairs"]) == 3
              and old["orders"] == [["cli", "nvml"], ["nvml", "cli"], ["cli", "nvml"]]}
    records = {}
    snapshots = output / "executed_sources"
    snapshots.mkdir()
    source_rows = {}
    input_rows = {}
    unique_windows = {}
    linux_records = {}
    for row in old["sources"] + generic["sources"]:
        source_rows[row["path"]] = row
    actual_comparisons = []
    for trial in range(3):
        compared = compare_pair(OLD / f"c1/t{trial}/b", OLD / f"c1/t{trial}/c")
        if not compared["all_protected"] or len(compared["records"]) != 24:
            raise AssertionError("A saved full24/48 scientific pair changed")
        actual_comparisons.append(compared)
        for label in ("b", "c"):
            side = read(OLD / f"c1/t{trial}/{label}/RESULT.json")
            node = read(OLD / f"c1/t{trial}/n{label}/NODE.json")
            source_rows.update({row["path"]: row for row in side["sources"] + node["monitoring"]["sources"]})
            for row in side["inputs"]:
                if row["path"] in input_rows and input_rows[row["path"]] != row:
                    raise AssertionError("The three complete pairs used different frozen inputs")
                input_rows[row["path"]] = row
            copied = OLD / f"c1/t{trial}/n{label}/OWNED_CONTROLLER_SOURCE.py"
            if data_identity(copied)["sha256"] != node["source"]["sha256"]:
                raise AssertionError("A timed original controller source snapshot changed")
            if node["status"] != "completed" or node["live_Windows_after_wait"] or not node["monitoring"]["library_closed"]:
                raise AssertionError("A preserved original node was not closed")
            raw_phase = node["monitoring"]["setup_seconds"] + sum(node["monitoring"]["query_seconds"]) + node["monitoring"]["shutdown_seconds"] + node["monitoring"]["final_source_guard_seconds"]
            mode = "cli" if label == "b" else "nvml"
            if raw_phase != old["pairs"][trial]["raw_CPU_phase_seconds"][mode] or node["complete_node_seconds"] != old["pairs"][trial]["raw_whole_seconds"][mode]:
                raise AssertionError("The actual monitoring or whole window differs from its original node")
            for row in node["guard"].get("owned_process_identities", []) + [node["owner"]]:
                unique_windows[(row["pid"], row["creation_time"])] = row
            for row in node["Linux_resource_receipts"]:
                linux_records[(row["pid"], row["creation_time"])] = row
    checks["all_three_saved_complete24_48_contents_recompared_exact"] = all(row["all_protected"] for row in actual_comparisons)
    checks["all_original_full_monitor_and_whole_windows_recomputed"] = True
    for label in ("g0", "g1"):
        row = read(OLD / label / "WSL_RESOURCE.json")
        linux_records[(row["pid"], row["creation_time"])] = row
    row = read(OWN / "g0/WSL_RESOURCE.json")
    linux_records[(row["pid"], row["creation_time"])] = row
    recovery = read(OWN / "c0/RESULT.json")
    fault = read(OWN / "f1/RESULT.json")
    checks["actual_interface_fault_and_closed_owned_worker8_guards"] = len(fault["checks"]) == 8 and all(fault["checks"].values())
    checks["actual_fresh_complete24_48_recovery7_guards"] = len(recovery["checks"]) == 7 and all(recovery["checks"].values())
    checks["recovery_actual_scientific_outputs_recompared_exact"] = compare_pair(OLD / "c1/t0/b", OWN / "c0/task")["all_protected"]
    for label in ("f1", "c0"):
        binding = read(OWN / label / "PRIVATE_CONTROLLER_BINDING.json")
        if data_identity(binding["actual_executed_main"]["path"]) != binding["actual_executed_main"]:
            raise AssertionError("The actual private controller source changed")
        checks[label + "_only_fault_census_changed_all_normal_AST_paths_exact"] = expected_fault_only_main(Path(binding["actual_executed_main"]["path"]))
        source_rows.update({row["path"]: row for row in read(OWN / label / "RESULT.json")["sources"]})
        node = read(OWN / label / "node/NODE.json")
        for row in node["guard"].get("owned_process_identities", []) + [node["owner"]]:
            unique_windows[(row["pid"], row["creation_time"])] = row
        for row in node["Linux_resource_receipts"]:
            linux_records[(row["pid"], row["creation_time"])] = row
    source_rows[str(Path(__file__))] = data_identity(Path(__file__))
    for index, row in enumerate(source_rows.values()):
        if data_identity(row["path"]) != row:
            raise AssertionError("A protected current source or native DLL changed")
        path = Path(row["path"])
        if path.suffix.lower() == ".py":
            shutil.copyfile(path, snapshots / (str(index) + "_" + path.name))
    for row in input_rows.values():
        if data_identity(row["path"]) != row:
            raise AssertionError("A protected complete original database, map or manifest changed")
    checks["all_current_scientific_sources_DLL_inputs_and_executed_snapshots_exact"] = bool(source_rows) and len(input_rows) == 9
    old_names = ("PROBE_LEASE_001.json", "PROBE_LEASE_002.json", "GENERIC_LEASE_001.json", "GENERIC_LEASE_002.json", "CAMPAIGN_LEASE_001.json", "CAMPAIGN_LEASE_002.json")
    lease_rows = []
    for folder, names in ((OLD, old_names), (OWN, tuple(label + "_lease.json" for label in ("g0", "f0", "f1", "c0")))):
        for name in names:
            value = read(folder / name)
            if value["status"] != "child_exited" or not value.get("lease_released_utc"):
                raise AssertionError("An actual original or supplemental lease remains open")
            lease_rows.append({"path": str(folder / name), "child_seconds": value["child_elapsed_seconds"], "returncode": value["returncode"]})
            for role in ("guard", "child"):
                row = {"pid": value[role + "_pid"], "creation_time": value[role + "_creation_time"]}
                unique_windows[(row["pid"], row["creation_time"])] = row
    for label in ("g0", "f0", "f1", "c0"):
        for row in read(OWN / (label + "_process.json"))["owned_process_identities"]:
            unique_windows[(row["pid"], row["creation_time"])] = row
    failed = read(OWN / "f0/node/NODE.json")
    for row in failed["live_Windows_after_wait"]:
        unique_windows[(row["pid"], row["creation_time"])] = row
    checks["original_first_fault_race_and_validated_two_worker_drain_preserved"] = len(failed["live_Windows_after_wait"]) == 2 and len(read(OWN / "f0/OWNED_FAILURE_DRAIN_001.json")["rows"]) == 2
    checks["all_saved_actual_owned_Windows_instances_gone_now"] = not remaining(list(unique_windows.values()))
    linux = [linux_identity_check(row) for row in linux_records.values()]
    checks["all_actual_old_and_new_Linux_instances_gone_now"] = len(linux) == 10 and all(not row["alive"] for row in linux)
    session = NativeMemory()
    try:
        try:
            session.read("unsupported")
        except ValueError:
            checks["unsupported_API_version_explicitly_rejected"] = True
        else:
            checks["unsupported_API_version_explicitly_rejected"] = False
    finally:
        session.close()
    checks["fresh_unsupported_check_native_session_shutdown"] = session.closed
    baseline_fee = sum(read(OLD / name)["child_elapsed_seconds"] for name in old_names[:-1]) + sum(row["complete_compare_seconds"] for row in old["pairs"])
    if baseline_fee != old["complete_first_fees"]:
        raise AssertionError("The original first/failed/generic/comparison fee ledger changed")
    raw_sides = sum(sum(row["raw_whole_seconds"].values()) for row in old["pairs"])
    residue = read(OLD / "CAMPAIGN_LEASE_002.json")["child_elapsed_seconds"] - raw_sides - sum(row["complete_compare_seconds"] for row in old["pairs"])
    if residue < 0:
        raise AssertionError("The whole original formal parent residue is negative")
    new_fee = sum(read(OWN / (label + "_lease.json"))["child_elapsed_seconds"] for label in ("g0", "f0", "f1", "c0"))
    measured_drain_command_seconds = .2301556
    fee_before_review = baseline_fee + old_close["final_review_seconds"] + residue + new_fee + measured_drain_command_seconds
    elapsed = time.perf_counter() - STARTED
    all_fee = fee_before_review + elapsed
    phases = {mode: statistics.median(row["raw_CPU_phase_seconds"][mode] + all_fee / 6 for row in old["pairs"]) for mode in ("cli", "nvml")}
    wholes = {mode: statistics.median(row["raw_whole_seconds"][mode] + all_fee / 6 for row in old["pairs"]) for mode in ("cli", "nvml")}
    reduction = 1 - phases["nvml"] / phases["cli"]
    checks["V4_complete_phase_reduction_at_least5percent_with_all_known_fees"] = reduction >= .05
    checks["V4_whole_median_at_most1percent_more_not_claimed_faster"] = wholes["nvml"] <= 1.01 * wholes["cli"]
    checks["V4_each_whole_at_most_original1_1"] = all(row["raw_whole_seconds"]["nvml"] + all_fee / 6 <= 1.1 * (row["raw_whole_seconds"]["cli"] + all_fee / 6) for row in old["pairs"])
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks["parent23_and_R122_unique_not_previously_accepted"] = state["additional_accepted_count"] == 23 and len(set(state["accepted_mechanisms"])) == 23 and not any(name.startswith("R122") for name in state["accepted_mechanisms"])
    checks["owned_output_and_disk_budget_preserved"] = sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) < 30 * 1024**3 and shutil.disk_usage(OWN).free > 20 * 1024**3
    if not all(checks.values()):
        atomic_json(output / "FAILED_CHECKS.json", checks)
        raise AssertionError("A V4 source, scientific, resource, recovery or complete cost check failed")
    result = {"status": "all_V4_checks_passed_final_exit_and_cost_pending", "checks": checks,
              "review_kind": "separate_root_self_review_no_subagent", "parent_count_before_acceptance": 23,
              "new_optimization_count": 0, "sources_and_inputs_checked": list(source_rows.values()) + list(input_rows.values()),
              "source_snapshots": len(list(snapshots.iterdir())), "actual_three_complete_pair_comparisons": actual_comparisons,
              "owned_Windows_process_records": list(unique_windows.values()), "actual_Linux_post_exit": linux,
              "old_and_new_released_leases": lease_rows, "fee_before_this_review": fee_before_review,
              "old_complete_formal_parent_unallocated_residue_seconds": residue,
              "manual_failed_worker_drain_command_seconds": measured_drain_command_seconds,
              "review_internal_seconds": elapsed, "preliminary_all_known_fee_seconds": all_fee,
              "complete_phase_medians_seconds": phases, "complete_phase_reduction": reduction,
              "complete_whole_medians_seconds": wholes, "complete_whole_ratio": wholes["nvml"] / wholes["cli"],
              "review_pid": os.getpid(), "review_creation_time": psutil.Process().create_time(),
              "scope": "optional_resource_monitor_fixed_original_R90_R92_R94_R10224query_consumer",
              "formal_three_pairs_reused": True, "fresh_three_performance_pairs_this_V4": False,
              "production_modified": False, "new_training_or_GT_or_unseen_object": False}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "phase_reduction": reduction,
                      "whole_ratio": result["complete_whole_ratio"]}), flush=True)


if __name__ == "__main__":
    main()
