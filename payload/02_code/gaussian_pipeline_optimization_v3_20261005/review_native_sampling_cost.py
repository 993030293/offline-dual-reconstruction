"""Audit complete sampling pairs, historical costs, sources and actual process exits."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001"
OLD = OWNED.with_name("rm_gpu001")
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare
from run_native_sampling_campaign import exact_rows


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    atomic_json(args.output/"PROCESS_IDENTITY.json", {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()})
    campaign = read(OWNED/"c1/RESULT.json")
    checks = {"three_fresh_current_complete_pairs": campaign["status"] == "three_complete_pairs_preserved_complete_cost_pending" and len(campaign["pairs"]) == 3,
              "all_registered_pair_guards_passed": all(all(p["checks"].values()) for p in campaign["pairs"]),
              "all43_complete_current_pixel_camera_PNG_fixtures": all(read(OWNED/"g1/RESULT.json")["checks"].values()) and len(read(OWNED/"g1/RESULT.json")["rows"]) == 43,
              "all21_actual_input_ownership_error_unload_guards": all(read(OWNED/"g2/RESULT.json")["checks"].values()) and len(read(OWNED/"g2/RESULT.json")["checks"]) == 21}
    sources = campaign["sources"]+read(OWNED/"g1/RESULT.json")["sources"]+read(OWNED/"g2/RESULT.json")["sources"]
    guards, node_results = [], []
    for trial in range(3):
        folder = OWNED/"c1"/f"t{trial}"
        entries = {}
        for mode, label in (("native", "n"), ("native_DLL", "d")):
            result = read(folder/label/"RESULT.json")
            node_results.append(result)
            sources += result["sources"]+result["inputs"]
            checks[f"pair{trial}_{mode}_complete_current_R120_R124_guards"] = all(result["checks"].values()) and all(result["current_consumer_result"]["checks"].values())
            guard = read(folder/(mode+"_GUARD.json"))
            guards.append(guard)
            checks[f"pair{trial}_{mode}_bounded_entry_completed"] = guard["status"] == "completed" and guard["returncode"] == 0 and guard["peak_tree_RSS_bytes"] <= 8*1024**3 and guard["elapsed_seconds"] <= 450
            restoration = read(folder/label/"SAMPLING_BINDING_RESTORATION.json")
            checks[f"pair{trial}_{mode}_live_bindings_restored"] = all(restoration.values())
            entries[mode] = Path(result["localization"]).parent.parent
        protection = compare(entries["native"], entries["native_DLL"], entries["native"]/"NEIGHBOR_TRACE.json", entries["native_DLL"]/"NEIGHBOR_TRACE.json")
        checks[f"pair{trial}_all_pixels_features_matches_and_geometry_rechecked"] = protection["all_protected"]
        checks[f"pair{trial}_all24_pose_centers_and_stats_exact_rechecked"] = exact_rows(entries["native"], entries["native_DLL"])["all24_registered_members_exact"]
    checks["all_current_sources_original_inputs_and_DLL_unchanged"] = all(row == data_identity(Path(row["path"])) for row in sources)
    build = read(OWNED/"b1/RESULT.json")
    checks["actual_DLL_and_strict_float_build_identity_same"] = data_identity(Path(build["DLL"]["path"])) == build["DLL"] and "/fp:strict" in build["command"] and build["sources"][1] == data_identity(CODE/"native_camera_sampling.cpp")
    first_names = ("B0_LEASE.json", "B1_LEASE.json", "G0_LEASE.json", "G1_LEASE.json", "G2_LEASE.json", "C0_LEASE.json")
    first = [read(OWNED/name) for name in first_names]
    campaign_lease = read(OWNED/"C1_LEASE.json")
    leases = first+[campaign_lease]
    checks["all_seven_actual_leases_released"] = all(row.get("lease_released_utc") for row in leases)
    failure = read(OWNED/"c0/FAILURE.json")
    failure_log = (OWNED/"c0/t0/native_DLL.log").read_text(encoding="utf-8")
    checks["first_build_and_260_character_path_failures_retained"] = first[0]["returncode"] == first[-1]["returncode"] == 1 and failure["status"] == "failed_verification" and "FileNotFoundError" in failure_log and (OWNED/"b0/source_snapshot").is_dir()
    checks["failed_candidate_callback_and_consumer_restored"] = all(read(OWNED/"c0/t0/native_DLL/SAMPLING_BINDING_RESTORATION.json").values())
    linux_receipts = [read(p) for p in OWNED.rglob("WSL_RESOURCE.json")]
    linux = {(r["pid"], r["creation_time"]): r for r in linux_receipts}
    checks["all_eight_saved_Linux_resource_guards_closed"] = len(linux) == 8 and all(r["closed"] and r["peak_RSS_bytes"] <= 32*1024**3 for r in linux_receipts)
    payload = json.dumps([{"pid": pid, "creation_time": born} for pid, born in linux])
    code = "import json,psutil; rows=json.loads("+repr(payload)+"); result=[]\n"
    code += "for r in rows:\n alive=False\n try:\n  p=psutil.Process(r['pid']); alive=p.create_time()==r['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n result.append(dict(r,alive=alive))\nprint(json.dumps(result))"
    actual_linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    checks["actual_Linux_PID_creation_time_instances_gone"] = len(actual_linux) == len(linux) and not any(r["alive"] for r in actual_linux)
    windows = {}
    extra = [read(p) for p in OWNED.rglob("PROCESS_IDENTITY.json") if not p.is_relative_to(args.output)]
    extra += [read(p) for p in OWNED.rglob("*_GUARD.json")]
    for record in [*leases, *guards, *node_results, *extra]:
        for row in walk(record):
            for pid, born in ((row.get("pid"), row.get("creation_time")), (row.get("guard_pid"), row.get("guard_creation_time")), (row.get("child_pid"), row.get("child_creation_time"))):
                if pid is not None and born is not None and (pid, born) not in linux:
                    windows[pid, born] = {"pid": pid, "creation_time": born}
    actual_windows = []
    for row in windows.values():
        alive = False
        try:
            process = psutil.Process(row["pid"])
            alive = process.create_time() == row["creation_time"] and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        actual_windows.append(dict(row, alive=alive))
    checks["actual_Windows_PID_creation_time_instances_gone"] = not any(r["alive"] for r in actual_windows)
    old = read(OLD/"NEGATIVE_CLOSURE_001.json")
    old_failure = read(OLD/"NEGATIVE_CLOSURE_ATTEMPT_001.json")
    old_text = read(OLD/"DOCUMENT_TEXT_REVIEW_001.json")
    checks["old_negative_cost_and_failed_closure_preserved"] = old["new_optimization_count"] == 0 and old_failure["status"] == "failed_closure" and old["known_nonoverlapping_total_before_this_closure_seconds"] >= 148.204
    formal_windows = sum(g["elapsed_seconds"] for g in guards)
    outside = campaign_lease["child_elapsed_seconds"]-formal_windows
    checks["formal_guard_windows_not_double_counted"] = outside >= 0
    fees = {"old_original_cpu_GPU_and_old_root_review": old["known_nonoverlapping_total_before_this_closure_seconds"],
            "old_failed_closure": old_failure["closure_before_publication_seconds"],
            "old_final_closure": old["closure_before_publication_seconds"],
            "old_document_text_review": old_text["actual_text_validation_seconds"],
            "new_first_failed_build_pixel_boundary_and_path_failure": sum(row["child_elapsed_seconds"] for row in first),
            "fresh_campaign_outside_six_guard_windows": outside,
            "current_root_review_before_publication": time.perf_counter()-started}
    allocation = sum(fees.values())/6
    medians = {key: {mode: value+allocation for mode, value in rows.items()} for key, rows in campaign["raw_medians"].items()}
    upper = {key: 1-rows["native_DLL"]/rows["native"] for key, rows in medians.items()}
    checks["even_measured_cost_lower_bound_stage_below5percent"] = upper["complete_reference_preparation_stage_seconds"] < .05
    checks["whole_median_nonincrease_without_quality_relaxation"] = upper["whole_seconds"] >= 0
    atomic_json(args.output/"REVIEW.json", {"status": "reviewed_cost_negative" if all(checks.values()) else "failed_verification",
        "review_kind": "separate_root_self_review_no_subagents", "independent_agent_review": False,
        "checks": checks, "current_identity_records": len(sources), "actual_Windows": actual_windows,
        "actual_Linux": actual_linux, "seven_completed_lease_paths": list(first_names)+["C1_LEASE.json"],
        "source_identities": sources, "raw_medians": campaign["raw_medians"], "raw_reductions": campaign["raw_reductions"],
        "measured_nonoverlapping_fee_lower_bound_seconds": fees, "each_side_measured_fee_lower_bound": allocation,
        "medians_with_measured_fee_lower_bound": medians, "improvement_upper_bounds_with_required_cost": upper,
        "remaining_post_exit_and_publication_cost_is_nonnegative_and_cannot_rescue_speed_gate": True,
        "new_optimization_count": 0, "whole_review_before_publication_seconds": time.perf_counter()-started,
        "not_run": ["conditional_public_component_after_positive_final_cost", "full1016_training", "Bitmap_read_metadata_export", "GT_or_unseen_object_whole_pipeline", "deployment"]})
    print(json.dumps({"checks": len(checks), "passed": all(checks.values()), "cost_reduction_upper": upper, "new_optimization_count": 0}), flush=True)
    if not all(checks.values()):
        raise AssertionError(json.dumps({k: v for k, v in checks.items() if not v}))


if __name__ == "__main__":
    main()
