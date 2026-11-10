"""Separately review full composition, real errors and actually drained owners."""

import argparse
import json
import msvcrt
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import psutil

from equal_fee_cost_bound import equal_fee_cost_bound
from run_q68_fused_campaign import compare
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-syntax-command-seconds", type=float, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = OUT / "r0"
    output.mkdir(exist_ok=False)
    parent = read(DOC / "CURRENT_STATE.json")
    formal = read(OUT / "c0/RESULT.json")
    interface, recovery = read(OUT / "v0_interfaces/RESULT.json"), read(OUT / "v1/RESULT.json")
    comparisons = [compare(OUT / "c0" / ("t" + str(i)) / "control", OUT / "c0" / ("t" + str(i)) / "combo") for i in range(3)]
    nodes = [read(OUT / "c0" / ("t" + str(i)) / (mode + "_node/NODE.json")) for i in range(3) for mode in ("control", "combo")]
    nodes += [read(OUT / "v0_node/NODE.json")]
    nodes += [read(OUT / "v1" / (mode + "_node/NODE.json")) for mode in ("fault_control", "fault_combo", "recovery_combo")]
    windows, linux, leases = [], [], {}
    for node in nodes:
        windows += [node["owner"], *node["guard"]["owned_process_identities"]]
        linux += node["actual_Linux_after_wait"]
    for label, expected in (("c0_first", 1), ("c0_valid_first", 0), ("c0_rest", 0), ("v0", 0), ("v1", 0)):
        lease = read(OUT / (label + "_lease.json"))
        if lease["status"] != "child_exited" or lease["returncode"] != expected or not lease.get("lease_released_utc"):
            raise AssertionError("An actual measured composition lease did not release")
        leases[label] = lease
        windows += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    windows = list({(row["pid"], row["creation_time"]): row for row in windows}.values())
    linux = list({(row["pid"], row["creation_time"]): row for row in linux}.values())
    script = "import json,psutil; rows=json.loads(" + repr(json.dumps(linux)) + "); result=[]\n"
    script += "for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n result.append(dict(row,actual_current_alive=alive))\nprint(json.dumps(result))"
    linux_now = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", script], text=True, timeout=30))
    formal_actual = sum(row["actual_spawn_to_exit_seconds"] for pair in formal["pairs"] for row in pair["rows"].values())
    formal_actual += sum(pair["actual_compare_seconds_charged_each_side"] for pair in formal["pairs"])
    tail = leases["c0_valid_first"]["child_elapsed_seconds"] + leases["c0_rest"]["child_elapsed_seconds"] - formal_actual
    if tail < 0 or args.validation_syntax_command_seconds < 0:
        raise AssertionError("The formal execution envelope or actual fee became invalid")
    fee = formal["initial_necessary_integration_fee_seconds"] + tail + args.validation_syntax_command_seconds
    fee += leases["v0"]["child_elapsed_seconds"] + leases["v1"]["child_elapsed_seconds"]
    native, combo = (formal["raw_complete_stage_medians"][key] for key in ("control", "combo"))
    bound = equal_fee_cost_bound(native, combo, fee)
    original = read(OUT / "c0/t0/control/RESULT.json")
    sources_same = formal["sources"] == [data_identity(row["path"]) for row in formal["sources"]]
    copies_same = all(data_identity(OUT / "c0/source_snapshot" / (f"{i:03d}_" + Path(row["path"]).name))["sha256"] == row["sha256"] for i, row in enumerate(formal["sources"]))
    inputs_same = all({**data_identity(row["path"].replace("/mnt/c/", "C:/")), "path": row["path"]} == row for row in original["inputs"])
    phases = {}
    for mode in ("control", "combo"):
        results = [read(OUT / "c0" / ("t" + str(i)) / mode / "RESULT.json") for i in range(3)]
        phase_rows = [read(OUT / "c0" / ("t" + str(i)) / mode / "IDENTITY_TRACE.json")["phases"] for i in range(3)]
        phases[mode] = {key: statistics.median(row[key] for row in results) for key in (
            "Scene_load_seconds", "all_pixel_and_state_check_seconds", "full_process_tree_peak_RSS_bytes", "peak_CUDA_reserved_bytes")}
        phases[mode]["all_four_file_identity_phase_seconds"] = statistics.median(sum(p["actual_phase_seconds_not_whole_acceptance"] for p in rows) for rows in phase_rows)
    checks = {
        "parent29_unique29_receipts29_and_combination_not_recounted": parent["additional_accepted_count"] == 29
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == len(parent["acceptance_receipts"]) == 29
            and parent["Q68_R153_new_optimization_count"] == 0 and parent["R153_algorithm_accepted"],
        "uniform3_whole_original_protocol_unchanged": parent["V5_minimum_complete_stage_net_reduction"] == .03 and parent["V5_maximum_whole_median_ratio"] == 1.01 and parent["V5_maximum_each_whole_ratio"] == 1.1,
        "three_fresh_fixed_orders_full1016_all15_and2_full_files_exact": len(formal["pairs"]) == 3
            and [p["order"] for p in formal["pairs"]] == [["control", "combo"], ["combo", "control"], ["control", "combo"]]
            and all(all(row["checks"].values()) for row in comparisons),
        "all_four_original17_and1036_lists_before_after_order_and_owners_exact": all(row["checks"]["both_complete_identity_phase5_checks"] and row["checks"]["all_four_identity_lists_same_order"] for row in comparisons),
        "owned_interfaces10_and_all360_public_full_file_identity_passed": len(interface["checks"]) == 10 and all(interface["checks"].values()),
        "two_actual_native_identity17_errors_prefix16_and_fresh_full_recovery": len(recovery["rows"]) == 3 and all(recovery["checks"].values()) and all(all(row["checks"].values()) for row in recovery["rows"]),
        "current_complete1036_inputs_and_all_frozen_sources_exact": len(original["inputs"]) == 1036 and inputs_same and sources_same,
        "all_formal_physical_source_copies_preserved_exact": copies_same,
        "failed_initial_syntax_source_and_all_actual_fees_preserved": (OUT / "FAILED_WINDOWS_ENTRY_SOURCE_001.py").is_file() and formal["initial_necessary_integration_fees"]["actual_first_failed_entry_command_seconds"] == read(OUT / "FAILED_WINDOWS_ENTRY_COST_001.json")["actual_failed_command_seconds"],
        "no_parent_library_rebuild_or_paid_parent_validation_recount": formal["no_new_library_build_existing_parents_not_recounted"] and Path(formal["existing_parent_cost_receipt"]).is_file() and Path(formal["existing_Q68_acceptance_receipt"]).is_file(),
        "formal_windows_once_tail_and_all_known_new_validation_fees_included": tail >= 0 and fee > formal["initial_necessary_integration_fee_seconds"],
        "original_node_time_RSS_and_all_actual_workers_drained": all(node["guard"]["elapsed_seconds"] <= node["guard"]["maximum_seconds"] and node["guard"]["peak_tree_RSS_bytes"] <= node["guard"]["maximum_RSS_bytes"] and not node["live_Windows_after_wait"] and all(not row["alive"] for row in node["actual_Linux_after_wait"]) for node in nodes),
        "actual_saved_Windows_and_Linux_instances_closed_now": all(windows_closed(row) for row in windows) and all(not row["actual_current_alive"] for row in linux_now),
        "all_five_actual_leases_released_expected_exit0_or_initial1": len(leases) == 5,
        "complete_net_and_whole_gates_pass_with_all_known_fees": bound["signed_reduction_at_known_lower_fee"] >= .03 and combo / native <= 1.01 and max(formal["whole_ratios"]) <= 1.1,
        "development_file_evidence_not_pixel_GT_blind_or_whole_generalization": interface["public_already_used_development_not_GT_blind_or_whole_generalization"] and recovery["no_training_checkpoint_or_cross_process_cache_resume_claim"],
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED_REVIEW.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("The separately audited actual complete composition failed")
    with (OUT.parent / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as stream:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    shutil.copyfile(Path(__file__), output / "EXECUTED_REVIEW_SOURCE.py")
    process = psutil.Process()
    atomic_json(output / "RESULT.json", {"status": "full_composition_root16_passed_actual_review_exit_final_cost_and_publication_pending0",
        "checks": checks, "raw_complete_stage_medians": formal["raw_complete_stage_medians"], "raw_stage_reduction": formal["raw_stage_reduction"],
        "phase_and_resource_medians": phases, "formal_actual_covered_once_seconds": formal_actual, "formal_unallocated_tail_seconds": tail,
        "known_integration_fee_plus_registered10_reserve_before_review_exit_seconds": fee, "cost_bound_under_registered_incremental_execution_model": bound,
        "actual_lease_seconds": {label: lease["child_elapsed_seconds"] for label, lease in leases.items()},
        "Windows_instances": windows, "Linux_instances": linux_now, "OS_lock_actually_acquired_and_released": True,
        "own_process": {"pid": process.pid, "creation_time": process.create_time()},
        "source": data_identity(Path(__file__)), "internal_before_result_seconds": time.perf_counter() - started,
        "separate_root_self_review_not_independent_agent": True, "new_optimization_count": 0, "parent_count": 29,
        "unknown_text_research_not_assumed_zero_or_total_ROI": True,
        "stage_scope": "same_complete_existing_camera_Scene_and_pixel_state_verification_not_entire_training_or_SfM",
        "goal_remains_active": True, "production_default_changed": False})
    print({"checks": len(checks), "known_fee_plus_reserve": fee, "net": bound["signed_reduction_at_known_lower_fee"],
        "Windows": len(windows), "Linux": len(linux_now), "new": 0, "parent": 29})


if __name__ == "__main__":
    main()
