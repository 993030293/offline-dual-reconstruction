"""Separately audit complete frozen digest pairs, real recovery and actual fees."""

import json
import msvcrt
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from scene_transform_cost_node import ROOT, CODE
from review_scene_batch_negative import read, windows_closed
from run_fused_camera_digest_campaign import compare
from equal_fee_cost_bound import equal_fee_cost_bound
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"


def main():
    started = time.perf_counter()
    output = OUT / "r0"
    output.mkdir(exist_ok=False)
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    formal = read(OUT / "c0/RESULT.json")
    diagnostic, generic, interface, recovery = [read(OUT / path) for path in (
        "p0/FINGERPRINT_COST.json", "g0/RESULT.json", "v0_interfaces/RESULT.json", "v1/RESULT.json")]
    comparisons = [compare(OUT / "c0" / ("t"+str(index)) / "original", OUT / "c0" / ("t"+str(index)) / "fused") for index in range(3)]
    native, fused = (formal["raw_complete_stage_medians"][key] for key in ("original", "fused"))
    nodes = [read(OUT / (name+"_node/NODE.json")) for name in ("p0", "g0", "v0_interfaces")]
    nodes += [read(OUT / "c0" / ("t"+str(index)) / (mode+"_node/NODE.json")) for index in range(3) for mode in ("original", "fused")]
    nodes += [read(OUT / "v1" / (mode+"_node/NODE.json")) for mode in ("fault_original", "fault_fused", "recovery_fused")]
    windows, linux = [], []
    for node in nodes:
        windows += [node["owner"], *node["guard"]["owned_process_identities"]]
        linux += node["actual_Linux_after_wait"]
    leases = {}
    for label, expected in (("p0", 0), ("g0", 0), ("c0_first", 1), ("c0_valid_first", 0), ("c0_rest", 0), ("v0_interfaces", 0), ("v1", 0)):
        lease = read(OUT / (label+"_lease.json"))
        if lease["status"] != "child_exited" or lease["returncode"] != expected or not lease.get("lease_released_utc"):
            raise AssertionError("A frozen actual digest lease has not closed")
        leases[label] = lease
        windows += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                    {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    windows = list({(row["pid"], row["creation_time"]): row for row in windows}.values())
    linux = list({(row["pid"], row["creation_time"]): row for row in linux}.values())
    code = "import json,psutil; rows=json.loads("+repr(json.dumps(linux))+"); result=[]\n"
    code += "for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n result.append(dict(row,actual_current_alive=alive))\nprint(json.dumps(result))"
    linux_now = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    formal_actual = sum(row["actual_spawn_to_exit_seconds"] for pair in formal["pairs"] for row in pair["rows"].values())
    formal_actual += sum(pair["actual_compare_seconds_charged_each_side"] for pair in formal["pairs"])
    formal_tail = leases["c0_valid_first"]["child_elapsed_seconds"] + leases["c0_rest"]["child_elapsed_seconds"] - formal_actual
    if formal_tail < 0:
        raise AssertionError("The actual formal envelope did not cover its recorded stage windows")
    fee = formal["initial_known_family_fee_seconds"] + leases["v0_interfaces"]["child_elapsed_seconds"] + leases["v1"]["child_elapsed_seconds"] + formal_tail
    bound = equal_fee_cost_bound(native, fused, fee)
    canonical = read(OUT / "c0/t0/original/RESULT.json")
    sources_now = formal["sources"] == [data_identity(row["path"]) for row in formal["sources"]]
    archives = all(data_identity(OUT / "c0/source_snapshot" / (f"{index:03d}_"+Path(row["path"]).name))["sha256"] == row["sha256"]
                   for index, row in enumerate(formal["sources"]))
    inputs_now = all({**data_identity(row["path"].replace("/mnt/c/", "C:/")), "path": row["path"]} == row for row in canonical["inputs"])
    checks = {
        "parent28_unique28_no_R153_promoted_early": parent["additional_accepted_count"] == 28
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == len(parent["acceptance_receipts"]) == 28
            and not any(name.startswith("R153") for name in parent["accepted_mechanisms"]),
        "uniform3_whole_and_quality_protocol_unchanged": parent["V5_minimum_complete_stage_net_reduction"] == .03
            and parent["V5_maximum_whole_median_ratio"] == 1.01 and parent["V5_maximum_each_whole_ratio"] == 1.1,
        "original_complete1016_cost_diagnostic7_passed": len(diagnostic["checks"]) == 7 and all(diagnostic["checks"].values()),
        "arbitrary_bits_offsets_scalar_vector_four44_and13_checks": len(generic["checks"]) == 13 and all(generic["checks"].values()),
        "serial_fusion_effect_distinct_from_worker_only_file_hash": generic["component_medians"]["serial"] < generic["component_medians"]["original"],
        "all_owned_cache13_and_existing_public32_full_pixel_checks": len(interface["checks"]) == 13 and all(interface["checks"].values()),
        "actual_two_image17_prefix_errors_and_full1016_recovery": all(recovery["checks"].values())
            and len(recovery["rows"]) == 3 and all(all(row["checks"].values()) for row in recovery["rows"]),
        "fresh_three_fixed_orders_full15_fields_and2_files_recompared": len(formal["pairs"]) == 3
            and [pair["order"] for pair in formal["pairs"]] == [["original", "fused"], ["fused", "original"], ["original", "fused"]]
            and all(all(row["checks"].values()) for row in comparisons),
        "current_complete_inputs_and_all_frozen_algorithm_sources_exact": inputs_now and sources_now,
        "all_executed_formal_source_archives_exact": archives,
        "first_platform_failure_source_and_fee_preserved": "from profile_scene_fingerprint_cost import ROOT, CODE, OUT" in (OUT / "FAILED_WINDOWS_ENTRY_SOURCE_001.py").read_text(encoding="utf-8")
            and leases["c0_first"]["returncode"] == 1 and formal["initial_known_family_fees"]["actual_failed_Windows_entry_before_any_complete_node"] == leases["c0_first"]["child_elapsed_seconds"],
        "compile_cost_in_g0_once_and_formal_windows_not_double_charged": formal_tail >= 0
            and formal["initial_known_family_fees"]["actual_g0_build_generic_and_component_included_once"] == leases["g0"]["child_elapsed_seconds"],
        "all_original_resources_and_actual_workers_drained": all(node["guard"]["peak_tree_RSS_bytes"] <= node["guard"]["maximum_RSS_bytes"]
            and node["guard"]["elapsed_seconds"] <= node["guard"]["maximum_seconds"] and not node["live_Windows_after_wait"]
            and not any(row["alive"] for row in node["actual_Linux_after_wait"]) for node in nodes),
        "saved_Windows_instances_and_actual_Linux_instances_closed_now": all(windows_closed(row) for row in windows)
            and all(not row["actual_current_alive"] for row in linux_now),
        "all_seven_actual_leases_released_and_expected_exits": len(leases) == 7,
        "raw_and_full_fee_stage_math_and_whole_gates_pass": native == statistics.median(pair["rows"]["original"]["complete_stage_seconds"] for pair in formal["pairs"])
            and fused == statistics.median(pair["rows"]["fused"]["complete_stage_seconds"] for pair in formal["pairs"])
            and 1-fused/native == formal["raw_stage_reduction"] and bound["signed_reduction_at_known_lower_fee"] >= .03
            and fused/native <= 1.01 and max(formal["whole_ratios"]) <= 1.1,
        "owned_scope_format_and_public_evidence_limits_explicit": interface["untracked_external_pointer_writes_outside_owned_immutable_scope"]
            and interface["public_already_used_development_not_GT_blind_or_whole_generalization"],
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED_REVIEW.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("The separate frozen complete digest audit failed")
    with (OUT.parent / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as stream:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    shutil.copyfile(Path(__file__), output / "EXECUTED_REVIEW_SOURCE.py")
    fields = {}
    for mode in ("original", "fused"):
        values = [read(OUT / "c0" / ("t"+str(index)) / mode / "RESULT.json") for index in range(3)]
        fields[mode] = {key: statistics.median(row[key] for row in values) for key in (
            "Scene_load_seconds", "all_pixel_and_state_check_seconds", "full_process_tree_peak_RSS_bytes", "peak_CUDA_reserved_bytes")}
    atomic_json(output / "RESULT.json", {"status": "frozen_complete_digest_stage_root17_passed_final_publication_pending0", "checks": checks,
        "root_separate_self_review_not_independent_agent": True, "known_fee_plus_registered_reserve_before_review_exit_seconds": fee,
        "cost_bound_under_declared_fee_model": bound, "raw_complete_stage_medians": formal["raw_complete_stage_medians"],
        "raw_stage_reduction": formal["raw_stage_reduction"], "formal_actual_covered_once_seconds": formal_actual,
        "formal_unallocated_actual_tail_seconds": formal_tail, "actual_lease_seconds": {name: row["child_elapsed_seconds"] for name, row in leases.items()},
        "Windows_instances": windows, "Linux_instances": linux_now, "phase_and_resource_medians": fields,
        "OS_lock_actually_acquired_and_released": True, "source": data_identity(Path(__file__)),
        "internal_before_result_seconds": time.perf_counter()-started, "unknown_text_research_cost_not_assumed_zero_or_ROI": True,
        "stage_scope": "complete_existing_camera_Scene_load_and_pixel_state_verification_not_training_or_entire_SfM_pipeline",
        "v1_STATE_is_historical_step_journal_terminal_RESULT_and_actual_exit_verified": True,
        "new_optimization_count": 0, "parent_count": 28, "goal_remains_active": True})
    print({"checks": len(checks), "known_fee_plus_reserve": fee, "cost_bound": bound, "Windows": len(windows), "Linux": len(linux_now), "new": 0})


if __name__ == "__main__":
    main()
