"""Perform a separate root self-review of exact outputs, work and owner closure."""

from pathlib import Path
import json
import statistics
import subprocess
import sys
import time

import psutil

from run_open_image_handoff_campaign import OUT, compare
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity
from image_source_open_census import image_source_open_census


def windows_path(value):
    return Path("C:/" + value[7:]) if value.startswith("/mnt/c/") else Path(value)


def current_identity(row):
    actual = data_identity(windows_path(row["path"]))
    return all(actual[key] == row[key] for key in ("bytes", "sha256"))


def main():
    started = time.perf_counter()
    output = OUT / "r1"
    output.mkdir(exist_ok=False)
    for name in ("box_work", "public_work"):
        (output / name).mkdir()
    formal, component, public, recovery = [read(OUT / name) for name in ("c0/RESULT.json", "g0/RESULT.json", "g1/RESULT.json", "v3/RESULT.json")]
    dataset = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
    public_dataset = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace"
    checks = {}
    with image_source_open_census(output / "box_work", dataset), image_source_open_census(output / "public_work", public_dataset):
        reference = read(OUT / "c0/t0/control/RESULT.json")
        checks["all_current1036_frozen_inputs_full_bytes_and_order_preserved"] = len(reference["inputs"]) == 1036 and all(current_identity(row) for row in reference["inputs"])
        checks["all360_current_public_full_source_bytes_preserved"] = len(public["sides"][0]["inputs_before"]) == 360 and all(current_identity(row) for row in public["sides"][0]["inputs_before"])
        checks["all_formal_physical_sources_still_exact"] = formal["sources"] == [data_identity(row["path"]) for row in formal["sources"]]
        snapshot = OUT / "c0/source_snapshot"
        checks["all_formal_frozen_physical_source_copies_exact"] = all(data_identity(snapshot / (f"{index:03d}_" + Path(row["path"]).name))["sha256"] == row["sha256"] for index, row in enumerate(formal["sources"]))
        checks["all19_native_component_and7_public_checks_passed"] = len(component["checks"]) == 19 and len(public["checks"]) == 7 and all(component["checks"].values()) and all(public["checks"].values())
        checks["three_fresh_fixed_interleaved_complete_pairs_and_original15_fields_two_files_exact"] = len(formal["pairs"]) == 3 and formal["orders"] == [["control", "handoff"], ["handoff", "control"], ["control", "handoff"]] and all(all(compare(OUT / "c0" / ("t" + str(index)) / "control", OUT / "c0" / ("t" + str(index)) / "handoff")["checks"].values()) for index in range(3))
        checks["actual_prepare17_errors_and_fresh_full1016_recovery_exact"] = all(recovery["checks"].values()) and all(all(row["checks"].values()) for row in recovery["rows"]) and all(compare(OUT / "c0/t0/control", OUT / "v3/recovery_handoff")["checks"].values())
        checks["actual_partial_copies_deleted_original_failed_outputs_retained"] = read(OUT / "v3/FORCED_PARTIAL_DELETE.json")["recovery_target_absent_before_fresh_entry"] and all((OUT / "v3" / mode / "FAULT_PREFIX.json").is_file() for mode in ("fault_control", "fault_handoff"))
    pairs = formal["pairs"]
    checks["all_three_original_full_source_work_denominators_and25percent_reduction"] = all(pair["rows"]["control"]["source_image_native_read_opens"] == 4064 and pair["rows"]["handoff"]["source_image_native_read_opens"] == 3048 for pair in pairs)
    checks["raw_whole_median_and_each_original_gates_preserved"] = formal["raw_whole_medians"]["handoff"] / formal["raw_whole_medians"]["control"] <= 1.01 and max(formal["whole_ratios"]) <= 1.1
    checks["actual_installed_Pillow12point3_lifecycle_not_newest_assumption"] = read(OUT / "PILLOW_LIFECYCLE_001.json")["Pillow_version"] == "12.3.0" and read(OUT / "PILLOW_LIFECYCLE_001.json")["context_file_closed_core_kept"] and read(OUT / "PILLOW_LIFECYCLE_001.json")["explicit_close_destroys_core"]
    checks["both_validation_failures_and_original_sources_preserved"] = all((OUT / name).is_file() for name in ("v1/fault_control_node/NODE.json", "v2/fault_control/RESULT.json", "FAILED_VALIDATION_V1_SOURCES/profile_faulted_image_handoff.py", "FAILED_VALIDATION_V2_SOURCES/profile_faulted_image_handoff.py"))
    node_paths = list((OUT / "c0").glob("t*/*_node/NODE.json")) + list((OUT / "v1").glob("*_node/NODE.json")) + list((OUT / "v2").glob("*_node/NODE.json")) + list((OUT / "v3").glob("*_node/NODE.json"))
    nodes = [read(path) for path in node_paths]
    windows, linux = [], []
    for node in nodes:
        windows.append(node["owner"])
        windows.extend(node["guard"]["owned_process_identities"])
        linux.extend(node["Linux_resource_receipts"])
    leases = [OUT.parent / "shf001_preflight_lease.json", *[OUT / (name + "_lease.json") for name in ("g0", "c0_first", "c0_rest", "v1", "v2", "v3", "g1", "r0")]]
    for path in leases:
        row = read(path)
        windows.extend({"pid": row[key], "creation_time": row[key.replace("pid", "creation_time")]} for key in ("guard_pid", "child_pid"))
    checks["all_saved_native_Windows_instances_currently_closed"] = all(windows_closed(row) for row in windows)
    checks["all_original_node_guards_terminal_and_saved_native_linux_closure"] = len(nodes) == 11 and all(not node["live_Windows_after_wait"] and all(not row["alive"] for row in node["actual_Linux_after_wait"]) and node["guard"]["status"] in ("completed", "failed_child") for node in nodes)
    checks["all_nine_original_resource_leases_released"] = all(read(path)["lease_released_utc"] and read(path)["status"] in ("child_exited", "lease_or_child_failed") and read(path)["returncode"] in (0, 1) for path in leases)
    minimal_linux = [{"pid": row["pid"], "creation_time": row["creation_time"]} for row in linux]
    code = 'import json,os,pathlib,psutil; rows=json.loads(' + repr(json.dumps(minimal_linux)) + '); out=[]\n'
    code += 'for row in rows:\n alive=False\n try:\n  p=psutil.Process(row["pid"]); alive=p.create_time()==row["creation_time"] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n out.append(dict(pid=row["pid"],creation_time=row["creation_time"],alive=alive))\n'
    code += 'other=[p.info for p in psutil.process_iter(["pid","cmdline"]) if p.pid!=os.getpid() and any("python" in x.lower() for x in (p.info["cmdline"] or []))]; print(json.dumps(dict(instances=out,other_python=other,boot_id=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip())))'
    actual_linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    checks["all_saved_linux_instances_closed_and_no_other_python_worker"] = not actual_linux["other_python"] and all(not row["alive"] for row in actual_linux["instances"])
    checks["original_measured_GPU_and_Linux_RSS_bounds_preserved"] = all(read(OUT / "c0" / ("t" + str(index)) / mode / "RESULT.json")["peak_CUDA_reserved_bytes"] <= read(OUT / "c0" / ("t" + str(index)) / mode / "RESULT.json")["CUDA_budget_bytes"] for index in range(3) for mode in ("control", "handoff")) and all(row["peak_tree_RSS_bytes"] <= row["maximum_RSS_bytes"] for row in linux)
    output_bytes = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    checks["original_output_bound_preserved"] = output_bytes <= 12 * 1024 ** 3
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks["parent29_unpromoted_family_not_already_accepted"] = parent["additional_accepted_count"] == len(parent["accepted_mechanisms"]) == len(parent["acceptance_receipts"]) == 29 and not any(name.startswith("R156") for name in parent["accepted_mechanisms"]) and parent["R156_new_optimization_count"] == 0
    checks["V5_speed3percent_and_quality_protocol_unchanged"] = parent["V5_minimum_complete_stage_net_reduction"] == .03 and parent["V5_maximum_whole_median_ratio"] == 1.01 and parent["V5_maximum_each_whole_ratio"] == 1.1 and parent["execution_mode"] == "single_chat_no_subagents_or_side_chat_dispatch"
    generic_work = sum(side["work"]["successful_native_read_opens"] for case in component["cases"] for side in case["sides"])
    review_work = sum(read(output / scope / "SOURCE_IMAGE_OPEN_WORK.json")["successful_native_read_opens"] for scope in ("box_work", "public_work"))
    failed_review_work = sum(read(OUT / "r0" / scope / "SOURCE_IMAGE_OPEN_WORK.json")["successful_native_read_opens"] for scope in ("box_work", "public_work"))
    measured_extra_work = generic_work + public["actual_related_source_read_work"] + sum(row["source_native_read_opens"] for row in recovery["rows"]) + read(OUT / "v2/fault_control/SOURCE_IMAGE_OPEN_WORK.json")["successful_native_read_opens"] + review_work + failed_review_work
    inherited_budget = 16256
    net_work = 1 - (3048 + (measured_extra_work + inherited_budget) / 6) / (4064 + (measured_extra_work + inherited_budget) / 6)
    checks["net_complete_source_work_above10percent_after_all_measured_and_conservative_inherited_budget"] = net_work >= .10
    atomic_json(output / "RESULT.json", {"status": "separate_root_self_review_actual_checks_passed_cost_and_publication_pending0" if all(checks.values()) else "failed_verification",
        "checks": checks, "review_is_independent_agent": False, "sources": [data_identity(Path(__file__))],
        "actual_Windows_instances_checked": len(windows), "actual_Linux_instances_checked": len(linux), "actual_linux": actual_linux,
        "original_leases_checked": [str(path) for path in leases], "output_bytes": output_bytes,
        "measured_extra_source_read_work": measured_extra_work, "current_review_measured_work": review_work,
        "conservative_inherited_source_work_budget_not_direct_counter_measurement": inherited_budget,
        "inherited_budget_basis": "Two original complete source JPEG entries each budgeted twice the current directly counted native4064 opens; original fixed1016/1036 input phases and1016 main plus1016 worker calls retained. This is a conservative conditional source-level budget, not a historical FD census or total development work claim.",
        "net_complete_source_work_reduction_under_declared_six_side_model": net_work,
        "internal_before_result_seconds": time.perf_counter() - started, "own_process": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "new_optimization_count": 0, "parent_count": 29, "whole_speed_training_GT_or_energy_improvement_claim": False})
    print({"checks": checks, "net_work_reduction": net_work, "extra_measured_work": measured_extra_work, "new": 0})
    if not all(checks.values()):
        raise AssertionError("A separately executed root self-review check failed")


if __name__ == "__main__":
    main()
