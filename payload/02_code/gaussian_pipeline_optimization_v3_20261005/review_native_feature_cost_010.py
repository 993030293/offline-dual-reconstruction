"""Review exact source diagnostics and real process closure separately."""

from contextlib import closing
from pathlib import Path
import time

import psutil

from native_sfm_match_shards import data_identity, gpu, read, readonly, save

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup010"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"


def main():
    started = time.perf_counter()
    output = OWN / "r1"
    output.mkdir()
    p = OWN / "p2"
    result = read(p / "RESULT.json")
    protocol = read(p / "PROTOCOL.json")
    rows = read(p / "FULL_NATIVE_READ_ROWS.json")
    thumbnails = read(p / "FIXED32_THUMBNAIL_ROWS.json")
    logs = read(p / "ORIGINAL_COMPLETE_WRITER_ROWS.json")
    names = (OLD / "image_list.txt").read_text(encoding="utf-8").splitlines()
    guards, births = [], []
    for run in ("p0", "p1", "p2"):
        guard = read(OWN / (run + "_GUARD.json"))
        lease = read(OWN / (run + "_lease.json"))
        guards.append({"run": run, "status": guard["status"], "guard_returncode": guard["returncode"],
            "lease_returncode": lease["returncode"], "elapsed_seconds": guard["elapsed_seconds"],
            "lease_released_utc": lease.get("lease_released_utc")})
        identities = guard["owned_process_identities"] + [
            {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
        for row in identities:
            try:
                alive = psutil.Process(row["pid"]).create_time() == row["creation_time"]
            except psutil.NoSuchProcess:
                alive = False
            births.append(dict(row, run=run, same_birth_live=alive))
    with closing(readonly(OLD / "features.db")) as connection:
        keys = dict(connection.execute("SELECT images.name,keypoints.rows FROM images JOIN keypoints USING(image_id)").fetchall())
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    code = CODE / "audit_native_feature_read_010_v2.py"
    snapshot = p / "sources" / code.name
    text = code.read_text(encoding="utf-8")
    task = (ROOT / "research/2026-10-05-gaussian-optimization-round3-50/NATIVE_FEATURE_SOURCE_COST_010_V2_TASK.md").read_text(encoding="utf-8")
    binding = (p / "sources/bitmap.cc").read_text(encoding="utf-8")
    license_text = (p / "sources/COPYING.txt").read_text(encoding="utf-8")
    source_bitmap = (OWN / "p0/sources/bitmap.cc").read_text(encoding="utf-8")
    checks = {
        "actual_p2_guard_and_lease_exit0_and_all8_original_checks": guards[2]["guard_returncode"] == 0 and guards[2]["lease_returncode"] == 0 and guards[2]["status"] == "completed" and len(result["checks"]) == 8 and all(result["checks"].values()),
        "all12_owned_birth_records_closed_and_leases_released": len(births) == 12 and not any(row["same_birth_live"] for row in births) and all(row["lease_released_utc"] for row in guards),
        "p0_180second_guard_failure_and_only600_log_progress_preserved": guards[0]["status"] == "failed_controller" and guards[0]["guard_returncode"] == 15 and '"completed_original_images": 600' in (OWN / "p0.log").read_text() and not (OWN / "p0/NATIVE_PREPARATION_ROWS.json").exists(),
        "p1_copyright_check_failure_before_images_preserved": guards[1]["status"] == "failed_child" and read(OWN / "p1/FAILURE.json")["completed_read_count"] == 0 and read(OWN / "p1/BINDING_SOURCE_FETCH.json")["status"] == "failed",
        "all1002_read_rows_complete_order_dimensions_and_channels": len(rows) == 1002 and [row["index"] for row in rows] == list(range(1002)) and [row["name"] for row in rows] == names and all(row["dimensions"] == [1624, 1240] and row["channels"] == 1 for row in rows),
        "all1002_original_log_to_database_counts_typed_exact": {row["name"]: row["keypoints"] for row in logs} == keys and sum(keys.values()) == 1779059,
        "fixed32_thumbnail_indices_not_full1002_claim": len(thumbnails) == 32 and [row["index"] for row in thumbnails] == [round(index * 1001 / 31) for index in range(32)] and all(row["thumbnail_dimensions"] == [1600, 1222] for row in thumbnails) and result["p0_timed_out_and_full1002_thumbnail_not_completed"],
        "read_and_thumbnail_component_sums_exact": sum(row["native_read_seconds"] for row in rows) == result["read1002_seconds"] and sum(row["native_thumbnail_seconds"] for row in thumbnails) == result["thumbnail32_seconds"],
        "all_current_inputs_and_module_sources_unchanged": all(data_identity(row["path"]) == row for row in result["sources"] + result["inputs"] + protocol["full_images"]),
        "actual_code_snapshot_retained_exact_and_no_SIFT_execution": data_identity(code)["sha256"] == data_identity(snapshot)["sha256"] and "run_native_jobs(" not in text,
        "official_companion_license_and_original_binding_fulltext_retained": len(binding.splitlines()) == 187 and "Redistribution and use" in license_text and "Copyright (c), ETH Zurich and UNC Chapel Hill." in license_text,
        "official_binding_no_declared_GIL_release_and_native_gray_contract_rec709": "gil_scoped_release" not in binding and "call_guard" not in binding and ".2126f" in source_bitmap and ".7152f" in source_bitmap and ".0722f" in source_bitmap,
        "direct_task_text_records_denominators_failed_sources_and_count0": "完整1002" in task and "32张不是完整1002缩放" in task and "COPYING.txt" in task and "父36/50" in task,
        "unchanged_parent36of50_and_optimization_count0": state["additional_accepted_count"] == 36 and len(state["accepted_mechanisms"]) == 36 and len(state["acceptance_receipts"]) == 36 and result["new_optimization_count"] == 0,
    }
    review = {"status": "separate_root_self_review_not_independent_agent", "checks": checks,
        "birth_records": births, "guards": guards, "read1002_seconds": result["read1002_seconds"],
        "thumbnail32_seconds": result["thumbnail32_seconds"], "thumbnail32_median_seconds": result["thumbnail32_median_seconds"],
        "orientations": result["original_orientation_values"],
        "known_guard_elapsed_seconds_sum": sum(row["elapsed_seconds"] for row in guards),
        "root_review_elapsed_seconds": time.perf_counter() - started, "gpu": gpu(),
        "prior_root_reader_failure": {"tool": "a6e102", "exit_code": 1, "seconds": 2.4400646,
            "error": "KeyError accepted_receipts; actual original field is acceptance_receipts"},
        "new_optimization_count": 0, "parent_count": 36}
    save(output / "RESULT.json", review)
    print({"checks": checks, "read1002_seconds": review["read1002_seconds"], "thumbnail32_seconds": review["thumbnail32_seconds"], "new_count": 0})
    if not all(checks.values()):
        raise AssertionError("A separate source-cost root check failed")


if __name__ == "__main__":
    main()
