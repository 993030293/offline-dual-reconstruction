"""Record complete native feature differences without weakening acceptance."""

from contextlib import closing
import json
from pathlib import Path
import shutil
import time

import numpy as np
import psutil

from native_sfm_match_shards import columns, data_identity, gpu, quote, read, readonly, save, schemas

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sft011"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"


def compare_complete(left, right):
    counts, differences, details = {}, {}, []
    compared_blob_bytes = 0
    with closing(readonly(left)) as a, closing(readonly(right)) as b:
        tables = schemas(a)
        if tables != schemas(b):
            raise AssertionError("The negative native diagnostic has different schemas")
        names = dict(a.execute("SELECT image_id,name FROM images").fetchall())
        for table in tables:
            info = columns(a, table)
            if info != columns(b, table):
                raise AssertionError("The negative native diagnostic has different column declarations")
            primary = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]]
            order = primary or [row[1] for row in info]
            statement = "SELECT * FROM " + quote(table) + " ORDER BY " + ",".join(map(quote, order))
            ca, cb = a.execute(statement), b.execute(statement)
            count = different = 0
            while True:
                x, y = ca.fetchone(), cb.fetchone()
                if x is None and y is None:
                    break
                if x is None or y is None:
                    raise AssertionError("The negative native diagnostic has different row populations")
                compared_blob_bytes += sum(len(value) for value in x if isinstance(value, bytes))
                if x != y:
                    different += 1
                    differing = [info[index][1] for index in range(len(x)) if x[index] != y[index]]
                    record = {"table": table, "row_index": count, "columns": differing}
                    if table in ("keypoints", "descriptors"):
                        record["image_id"] = x[0]
                        record["image_name"] = names[x[0]]
                        data_index = next(index for index, column in enumerate(info) if column[1] == "data")
                        if differing == ["data"] and len(x[data_index]) == len(y[data_index]):
                            dtype = np.uint8 if table == "descriptors" else np.dtype("<f4")
                            first, second = np.frombuffer(x[data_index], dtype=dtype), np.frombuffer(y[data_index], dtype=dtype)
                            different_values = first != second
                            record["different_numeric_values"] = int(np.count_nonzero(different_values))
                            record["max_absolute_numeric_delta"] = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
                            record["first_different_flat_indices"] = np.flatnonzero(different_values)[:8].tolist()
                    if len(details) < 32:
                        details.append(record)
                count += 1
            counts[table] = count
            differences[table] = different
            ca.close(); cb.close()
    return {"all_columns_exact": not any(differences.values()), "table_row_counts": counts,
        "different_row_counts": differences, "first32_different_rows": details,
        "all_rows_directly_compared_BLOB_bytes": compared_blob_bytes,
        "not_a_relaxed_quality_gate_or_file_hash_oracle": True}


def main():
    started = time.perf_counter()
    output = OWN / "r2"
    output.mkdir()
    p = OWN / "g0"
    protocol = read(p / "PROTOCOL.json")
    failure = read(p / "FAILURE.json")
    baseline, candidate = read(p / "A/ENTRY.json"), read(p / "B/ENTRY.json")
    direct = compare_complete(baseline["database"], candidate["database"])
    original_baseline = compare_complete(OLD / "features.db", baseline["database"])
    original_candidate = compare_complete(OLD / "features.db", candidate["database"])
    guard, lease = read(OWN / "g0_GUARD.json"), read(OWN / "g0_lease.json")
    records = guard["owned_process_identities"] + [
        {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    for side in ("A", "B"):
        records += read(p / side / "NATIVE_PROCESSES.json")
    births = []
    for pid, birth in {(row["pid"], row["creation_time"]) for row in records}:
        try:
            live = psutil.Process(pid).create_time() == birth
        except psutil.NoSuchProcess:
            live = False
        births.append({"pid": pid, "creation_time": birth, "same_birth_live": live})
    text = (CODE / "native_feature_thread_transfer_011.py").read_text(encoding="utf-8")
    task = (ROOT / "research/2026-10-05-gaussian-optimization-round3-50/EXISTING_OIIO_FEATURE_TRANSFER_011_TASK.md").read_text(encoding="utf-8")
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    b_seconds, c_seconds = (entry["complete_metadata_to_ready_features_seconds"] for entry in (baseline, candidate))
    checks = {
        "actual_guard_failed_at_complete_quality_comparator_not_hidden": guard["status"] == "failed_child" and guard["returncode"] == 1 and lease["returncode"] == 1 and "descriptors row 417 columns data" in failure["error"] and not (p / "RESULT.json").exists(),
        "both_actual_single_native_CLI_processes_completed_exit0": all(read(p / side / "NATIVE_PROCESSES.json")[0]["returncode"] == 0 for side in ("A", "B")),
        "all1002_images_original_population_and_summary_counts_preserved": baseline["summary"] == candidate["summary"] and baseline["summary"]["image_count"] == 1002 and sum(candidate["summary"]["keypoint_counts"].values()) == 1779059,
        "all12_native_tables_scanned_directly_complete_in_three_comparisons": all(len(value["table_row_counts"]) == 12 and value["all_rows_directly_compared_BLOB_bytes"] == 270417032 for value in (direct, original_baseline, original_candidate)),
        "metadata_all_nonfeature_tables_exact": all(all(count == 0 for table, count in value["different_row_counts"].items() if table not in ("keypoints", "descriptors")) for value in (direct, original_baseline, original_candidate)),
        "actual_feature_byte_difference_kept_as_quality_failure": not direct["all_columns_exact"] and direct["different_row_counts"]["descriptors"] >= 1,
        "raw_complete_stage_slower_net2percent_not_reachable": c_seconds > b_seconds,
        "only_child_OIIO_control_changed_parent_not_mutated": baseline["native"]["environment_changes_only"] == {} and candidate["native"]["environment_changes_only"] == {"OPENIMAGEIO_THREADS": "1"} and 'os.environ[' not in text,
        "all12_owned_native_and_guard_births_closed_lease_released": len(births) == 12 and not any(row["same_birth_live"] for row in births) and bool(lease.get("lease_released_utc")),
        "all_original_sources_DLLs_inputs_and1002_images_still_exact": all(data_identity(row["path"]) == row for row in protocol["sources"] + protocol["inputs"] + protocol["images"]),
        "own_source_snapshot_exact": data_identity(CODE / "native_feature_thread_transfer_011.py")["sha256"] == data_identity(p / "sources/native_feature_thread_transfer_011.py")["sha256"],
        "existing_control_not_new_independent_mechanism": protocol["existing_R123_thread_control_transferred_not_new_mechanism"] and '不另造R167独立优化' in task,
        "formal_generic_public_error_cancel_full_recovery_not_run_after_stop": not any((OWN / name / "PROTOCOL.json").exists() for name in ("c1", "c2", "c3", "v0", "x0", "r1")) and not (OWN / "r1/A/ENTRY.json").exists(),
        "parent36of50_and36_acceptances_unchanged": state["additional_accepted_count"] == 36 and len(state["accepted_mechanisms"]) == 36 and len(state["acceptance_receipts"]) == 36,
        "registered_disk_and_output_budget_preserved": shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3 and sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 4 * 1024 ** 3,
    }
    result = {"status": "closed_negative_quality_bytes_differ_and_complete_stage_slower0",
        "review_kind": "separate_root_self_review_not_independent_agent", "checks": checks,
        "baseline_candidate_full_direct_difference": direct,
        "original_baseline_full_direct_difference": original_baseline,
        "original_candidate_full_direct_difference": original_candidate,
        "baseline_seconds": b_seconds, "candidate_seconds": c_seconds, "raw_whole_ratio": c_seconds / b_seconds,
        "failure_cause_environment_vs_native_nondeterminism_not_proven": True,
        "prior_root_failures_preserved": "r0/r1 FAILURE.json and SOURCE_FAILED.py; actual schema lookup and recovery protocol/entry checks",
        "known_g0_guard_seconds": guard["elapsed_seconds"], "lease_released_utc": lease.get("lease_released_utc"),
        "owned_birth_records": births, "root_count": 36, "new_optimization_count": 0,
        "elapsed_seconds": time.perf_counter() - started, "gpu": gpu()}
    save(output / "RESULT.json", result)
    print(json.dumps({"checks": checks, "ratio": result["raw_whole_ratio"], "B_C_differences": direct["different_row_counts"],
        "original_B_differences": original_baseline["different_row_counts"],
        "original_C_differences": original_candidate["different_row_counts"], "count": 0}))
    if not all(checks.values()):
        raise AssertionError("A separate native feature-transfer negative closure check failed")


if __name__ == "__main__":
    main()
