"""Close a quality-exact but slower full native grayscale-bank candidate."""

from pathlib import Path
import json
import shutil
import time

import psutil

from native_sfm_match_shards import compare_databases, data_identity, gpu, read, save

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sfg001"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"


def main():
    started = time.perf_counter()
    output = OWN / "r0"
    output.mkdir()
    p0 = read(OWN / "p0/RESULT.json")
    g0 = read(OWN / "g0/RESULT.json")
    protocol = read(OWN / "g0/PROTOCOL.json")
    guards, births = [], []
    for run in ("p0", "g0"):
        guard = read(OWN / (run + "_GUARD.json"))
        lease = read(OWN / (run + "_lease.json"))
        guards.append({"run": run, "returncode": guard["returncode"], "status": guard["status"],
            "elapsed_seconds": guard["elapsed_seconds"], "lease_returncode": lease["returncode"],
            "lease_released_utc": lease.get("lease_released_utc")})
        records = guard["owned_process_identities"] + [
            {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
        for side in ("A", "B"):
            records += read(OWN / run / side / "NATIVE_PROCESSES.json")
        records += read(OWN / run / "B/BANK.json")["rows"]
        unique = {(row["pid"], row["creation_time"]) for row in records}
        for pid, birth in unique:
            try:
                live = psutil.Process(pid).create_time() == birth
            except psutil.NoSuchProcess:
                live = False
            births.append({"run": run, "pid": pid, "creation_time": birth, "same_birth_live": live})
    a, b = OWN / "g0/A/database.db", OWN / "g0/B/database.db"
    direct = compare_databases(a, b)
    original = compare_databases(OLD / "features.db", b)
    baseline_seconds = g0["baseline"]["complete_metadata_to_ready_features_seconds"]
    candidate_seconds = g0["candidate"]["complete_metadata_to_ready_features_seconds"]
    code = CODE / "native_feature_gray_bank_001.py"
    text = code.read_text(encoding="utf-8")
    task = (ROOT / "research/2026-10-05-gaussian-optimization-round3-50/R166_EXACT_GRAY_BANK_TASK.md").read_text(encoding="utf-8")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    snapshot_ok = all(data_identity(OWN / run / "sources" / code.name)["sha256"] == data_identity(code)["sha256"] for run in ("p0", "g0"))
    checks = {
        "component32_and_full1002_original7_checks_all_passed": len(p0["checks"]) == 7 and len(g0["checks"]) == 7 and all(p0["checks"].values()) and all(g0["checks"].values()),
        "actual_two_guards_leases_exit0_released": all(row["status"] == "completed" and row["returncode"] == 0 and row["lease_returncode"] == 0 and row["lease_released_utc"] for row in guards),
        "all39_unique_per_run_owned_births_closed": len(births) == 39 and not any(row["same_birth_live"] for row in births),
        "independent_original_and_current_all12_native_column_values_exact": direct["complete_values_exact"] and original["complete_values_exact"] and direct["totals"]["tables"] == 12 and original["totals"]["directly_compared_BLOB_bytes"] == 270417032,
        "whole1002_keypoint_descriptor_population_and_names_exact": len(g0["baseline"]["summary"]["keypoint_counts"]) == 1002 and g0["baseline"]["summary"] == g0["candidate"]["summary"] and sum(g0["candidate"]["summary"]["keypoint_counts"].values()) == 1779059,
        "whole_original_bank_pixel_oracle_complete_not_hash": g0["pixel_oracle"]["images"] == 1002 and g0["pixel_oracle"]["directly_compared_pixel_bytes"] == 2017787520 and g0["pixel_oracle"]["whole_native_gray_pixels_exact"] and g0["pixel_oracle"]["not_a_file_hash_or_sampled_pixel_oracle"],
        "all_original_sources_inputs_and_full1002_JPEG_identity_unchanged": all(data_identity(row["path"]) == row for row in protocol["sources"] + protocol["inputs"] + protocol["images"]),
        "both_saved_algorithm_source_copies_exact": snapshot_ok,
        "cold_full_bank_startup_and_write_are_in_complete_candidate": g0["candidate"]["includes_seed_copy_delete_bank_startup_decode_write_drain_native_summary_publish"] and g0["candidate"]["bank"]["complete_cold_bank_seconds_including_pool_startup_and_shutdown"] > 0,
        "same_single_original_GPU_params_no_rescale_or_frame_reduction": 'bitmap.thumbnail(' not in text and 'max_workers=8' in text and g0["candidate"]["native"]["native_processes"] == 1 and not g0["candidate"]["component_only"],
        "full_stage_raw_slowdown_and_fixed2percent_gate_failed": candidate_seconds > baseline_seconds and (baseline_seconds - candidate_seconds) / baseline_seconds < .02,
        "stage_whole_limit_not_relaxed_and_gain_not_work_relabelled": g0["raw_whole_ratio"] <= 1.1 and '失败不改计工作量或I/O收益' in task and '净速度≥2%' in task,
        "remaining_formal_and_generic_public_error_full_recovery_not_run_after_stop": not any((OWN / name).exists() for name in ("c1", "c2", "c3", "v0", "x0", "r1")),
        "reported_consumer_is_declared_metadata_regeneration_not_empty_RAW": '不冒称第一次空DB从RAW创建身份' in task and 'seed_database(database)' in text,
        "parent36of50_acceptances_unchanged_and_no_new_count": state["additional_accepted_count"] == 36 and len(state["accepted_mechanisms"]) == 36 and len(state["acceptance_receipts"]) == 36 and g0["new_optimization_count"] == 0,
        "registered_output_and_disk_budget_preserved": sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 24 * 1024 ** 3 and shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3,
    }
    result = {"status": "closed_negative_raw_complete_stage_slower_quality_exact_no_formal_acceptance",
        "review_kind": "separate_root_self_review_not_independent_agent", "checks": checks,
        "direct_native_oracle": direct, "direct_original_candidate_oracle": original,
        "guards": guards, "owned_birth_records": births,
        "baseline_seconds": baseline_seconds, "candidate_seconds": candidate_seconds,
        "raw_whole_ratio": candidate_seconds / baseline_seconds,
        "raw_gain_upper_bound_before_positive_fees": (baseline_seconds - candidate_seconds) / baseline_seconds,
        "negative_stage_cannot_reach_net2percent_with_nonnegative_symmetric_fees": True,
        "known_original010_guard_seconds": read(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup010/r1/RESULT.json")["known_guard_elapsed_seconds_sum"],
        "known_R166_guard_seconds": sum(row["elapsed_seconds"] for row in guards),
        "new_optimization_count": 0, "root_count": 36, "elapsed_seconds": time.perf_counter() - started,
        "gpu": gpu()}
    save(output / "RESULT.json", result)
    print(json.dumps({"checks": checks, "raw_ratio": result["raw_whole_ratio"], "count": 0}))
    if not all(checks.values()):
        raise AssertionError("A separate negative exact-bank closure check failed")


if __name__ == "__main__":
    main()
