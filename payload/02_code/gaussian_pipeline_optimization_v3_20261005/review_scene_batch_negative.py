"""Audit actual terminal Scene attempts and preserve signed-cost label corrections."""

import json
import msvcrt
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from scene_transform_cost_node import ROOT, CODE, OUT
from equal_fee_cost_bound import equal_fee_cost_bound
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("A receipt contains a duplicate JSON key")
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def windows_closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    output = OUT / "r0"
    output.mkdir(exist_ok=False)
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    diagnostic, generic, regression = read(OUT / "p0/TRANSFORM_COST.json"), read(OUT / "g0/RESULT.json"), read(OUT / "g2/RESULT.json")
    old = read(OUT / "c0/STATE.json")
    old_trace = read(OUT / "c0/t0/batch/BATCH_TRACE.json")
    formal = read(OUT / "c1/RESULT.json")
    actual = formal["pairs"][0]
    node_paths = [OUT / (label + "_node/NODE.json") for label in ("p0", "g0", "g1", "g2")]
    node_paths += [OUT / branch / "t0" / (mode + "_node/NODE.json") for branch in ("c0", "c1") for mode in ("native", "batch")]
    nodes = [read(path) for path in node_paths]
    identities, linux = [], []
    for node in nodes:
        identities.extend([node["owner"], *node["guard"]["owned_process_identities"]])
        linux.extend(node["actual_Linux_after_wait"])
    leases, fees = {}, {}
    for label, expected in (("p0", 0), ("g0", 0), ("g1", 1), ("g2", 0), ("c0_first", 1), ("c1_first", 0)):
        lease = read(OUT / (label + "_lease.json"))
        if lease["status"] != "child_exited" or lease["returncode"] != expected or not lease.get("lease_released_utc"):
            raise AssertionError("An expected actual Scene lease did not close")
        leases[label], fees[label] = lease, lease["child_elapsed_seconds"]
        identities.extend([{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                           {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}])
    unique = {(row["pid"], row["creation_time"]): row for row in identities}
    linux_unique = {(row["pid"], row["creation_time"]): row for row in linux}
    code = "import json,psutil; rows=json.loads(" + repr(json.dumps(list(linux_unique.values()))) + "); result=[]\n"
    code += "for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n result.append(dict(row,actual_current_alive=alive))\nprint(json.dumps(result))"
    linux_now = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    old_sources_saved = all(data_identity(OUT / "c0/source_snapshot" / (f"{index:03d}_" + Path(row["path"]).name))["sha256"] == row["sha256"]
                            for index, row in enumerate(old["sources"]))
    current_sources = formal["sources"] == [data_identity(row["path"]) for row in formal["sources"]]
    covered = sum(row["complete_stage_seconds"] for row in actual["rows"].values()) - actual["actual_compare_seconds_charged_each_side"]
    tail = fees["c1_first"] - covered
    if tail < 0:
        raise AssertionError("The actual complete Scene envelope did not cover its stage windows")
    known = formal["initial_known_family_fee_lower_bound_seconds"] + tail
    n, c = (formal["raw_complete_stage_medians"][key] for key in ("native", "batch"))
    corrected = equal_fee_cost_bound(n, c, known)
    large_fee = equal_fee_cost_bound(n, c, known + 1e8)
    positive = equal_fee_cost_bound(44.04442879999988, 39.597734899958596, 655.3733049497881)
    positive_larger = equal_fee_cost_bound(44.04442879999988, 39.597734899958596, 655.3733049497881 + 1e8)
    checks = {
        "parent28_unique28_receipts_and_R152_not_accepted": state["additional_accepted_count"] == 28
            and len(set(state["accepted_mechanisms"])) == len(state["acceptance_receipts"]) == 28
            and not any(name.startswith("R152") for name in state["accepted_mechanisms"]),
        "uniform3percent_and_whole_gates_kept": state["V5_minimum_complete_stage_net_reduction"] == .03
            and state["V5_maximum_whole_median_ratio"] == 1.01 and state["V5_maximum_each_whole_ratio"] == 1.1,
        "original_complete_diagnostic11_and7_and_generic5_checks": all(read(OUT / "p0/RESULT.json")["checks"].values())
            and all(diagnostic["checks"].values()) and all(generic["checks"].values()),
        "old_c0_real_prefetch_bypass_failed_and_source_archive_preserved": old["status"] == "failed_original_complete_node"
            and old_trace["trace"]["batch_groups"] == 0 and old_trace["trace"]["native_calls"] == 1016 and old_sources_saved,
        "g1_API_reader_failure_saved_separately_from_algorithm": "has no attribute 'activate_reference'" in (OUT / "g1_node/child.log").read_text(encoding="utf-8")
            and "common.activate_reference" in (OUT / "g1/source_snapshot/check_batch_prefetch_regression.py").read_text(encoding="utf-8"),
        "real_g2_late_prefetch3_full_state_regression_passed": all(regression["checks"].values())
            and regression["traces"]["batch"]["batch"]["batch_prepared"] == 3,
        "current_c1_one_registered_full1016_pair_all15_fields_and2_files": len(formal["pairs"]) == 1
            and all(actual["comparison"]["checks"].values()) and actual["comparison"]["complete_camera_rows"] == 1016,
        "actual_c1_full1016_batch_prepared_consumed_and_no_native_fallback": all(read(OUT / "c1/t0/batch/BATCH_TRACE.json")["checks"].values()),
        "actual_complete_stage_slower_and_raw_binary64_arithmetic_exact": c > n
            and 1 - c / n == formal["raw_stage_reduction"],
        "negative_signed_lower_fee_value_is_not_strict_upper_bound": formal["complete_stage_reduction_upper_bound"] < 0
            and corrected["conservative_sample_reduction_upper_bound"] == large_fee["conservative_sample_reduction_upper_bound"] == 0
            and large_fee["signed_reduction_at_known_lower_fee"] > corrected["signed_reduction_at_known_lower_fee"],
        "positive_gain_bound_remains_monotonic_and_zero_case_exact": positive_larger["conservative_sample_reduction_upper_bound"]
            < positive["conservative_sample_reduction_upper_bound"] and equal_fee_cost_bound(1., 1., 100.)["conservative_sample_reduction_upper_bound"] == 0,
        "actual_formal_window_not_double_charged_all_failed_leases_kept": tail >= 0 and known > 137
            and fees["c0_first"] > 84 and fees["g1"] > 1,
        "all_source_snapshots_match_current_c1_physical_sources": current_sources,
        "all_Windows_and_Linux_saved_instances_actually_closed": all(windows_closed(row) for row in unique.values())
            and all(not row["actual_current_alive"] for row in linux_now),
        "all_original_node_resource_limits_and_closed_workers": all(node["guard"]["peak_tree_RSS_bytes"] <= node["guard"]["maximum_RSS_bytes"]
            and node["guard"]["elapsed_seconds"] <= node["guard"]["maximum_seconds"]
            and not node["live_Windows_after_wait"] and all(not row["alive"] for row in node["actual_Linux_after_wait"]) for node in nodes),
        "remaining_pairs_normal_errors_recovery_not_run_after_cost_failure": not (OUT / "c1/t1").exists()
            and not (OUT / "v0").exists(),
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED_REVIEW.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("A separate Scene negative review check failed")
    with (OUT.parent / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as stream:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for path in (Path(__file__), CODE / "equal_fee_cost_bound.py", CODE / "run_batched_camera_scene_campaign.py"):
        shutil.copyfile(path, snapshots / path.name)
    atomic_json(output / "RESULT.json", {"status": "closed_negative_one_full1016_pair_slower7point028427percent_new0",
        "checks": checks, "root_separate_self_review_not_independent_agent": True, "actual_family_lease_seconds": fees,
        "known_fee_lower_bound_before_review_exit_seconds": known, "actual_c1_formal_covered_once_seconds": covered,
        "actual_c1_formal_unallocated_tail_seconds": tail, "corrected_sample_cost_bound": corrected,
        "old_c1_signed_lower_fee_gain_field_not_valid_strict_upper_bound": True,
        "old_c1_raw_values_and_0_decision_unchanged": True, "Windows_instances": list(unique.values()), "Linux_instances": linux_now,
        "OS_lock_actually_acquired_and_released": True,
        "phase_scene_load_seconds": {mode: actual["rows"][mode]["native_Scene_load_seconds"] for mode in ("native", "batch")},
        "complete_stage_seconds": formal["raw_complete_stage_medians"], "raw_stage_reduction": formal["raw_stage_reduction"],
        "source_identity": data_identity(Path(__file__)), "internal_before_result_seconds": time.perf_counter() - started,
        "normal_errors_recovery_formal3_external_GT_training_production": "not_run",
        "unknown_text_research_and_publisher_fees_not_zero": True, "new_optimization_count": 0, "parent_count": 28,
        "goal_remains_active": True})
    print(json.dumps({"checks": len(checks), "corrected_cost_bound": corrected, "known_fee_lower_bound": known,
        "Windows_instances": len(unique), "Linux_instances": len(linux_now), "parent_count": 28, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
