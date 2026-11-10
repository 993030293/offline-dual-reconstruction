"""Run complete native matching in isolated shards and compare every database value."""
from collections import defaultdict
from contextlib import closing
import gc
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup009/R165"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"
sys.path.insert(0, str(V1))
from run_sfm_preflight import database_summary
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def readonly(path):
    return sqlite3.connect("file:" + Path(path).resolve().as_posix() + "?mode=ro", uri=True)


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def schemas(connection):
    return dict(connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall())


def columns(connection, table):
    rows = connection.execute("PRAGMA table_info(" + quote(table) + ")").fetchall()
    if not rows:
        raise ValueError("The native database table has no declared columns")
    return rows


def compare_databases(left, right, exclude=()):
    totals = {"tables": 0, "rows": 0, "directly_compared_BLOB_bytes": 0}
    counts = {}
    with closing(readonly(left)) as a, closing(readonly(right)) as b:
        tables = schemas(a)
        if tables != schemas(b):
            raise AssertionError("The complete native schemas differ")
        for table in tables:
            if table in exclude:
                continue
            info = columns(a, table)
            if info != columns(b, table):
                raise AssertionError("The complete native column declarations differ")
            primary = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]]
            order = primary or [row[1] for row in info]
            sql = "SELECT * FROM " + quote(table) + " ORDER BY " + ",".join(map(quote, order))
            ca, cb = a.execute(sql), b.execute(sql)
            count = 0
            while True:
                x, y = ca.fetchone(), cb.fetchone()
                if x != y:
                    differing = [info[i][1] for i in range(len(x or y)) if x is None or y is None or x[i] != y[i]]
                    raise AssertionError("Native table values differ at " + table + " row " + str(count) + " columns " + ",".join(differing))
                if x is None:
                    break
                count += 1
                totals["directly_compared_BLOB_bytes"] += sum(len(value) for value in x if isinstance(value, bytes))
            ca.close(); cb.close()
            counts[table] = count; totals["tables"] += 1; totals["rows"] += count
    return {"complete_values_exact": True, "counts": counts, "totals": totals, "direct_typed_values_not_hash_or_file_layout_oracle": True}


def partition_pairs(database, pair_lines):
    with closing(readonly(database)) as connection:
        names = dict(connection.execute("SELECT name,image_id FROM images").fetchall())
        keypoints = dict(connection.execute("SELECT image_id,rows FROM keypoints").fetchall())
    groups = defaultdict(list)
    seen = set()
    for index, line in enumerate(pair_lines):
        a, b = line.split()
        if a not in names or b not in names or a == b:
            raise ValueError("An original matching pair endpoint is invalid")
        edge = tuple(sorted((names[a], names[b])))
        if edge in seen:
            raise ValueError("The original undirected pair is duplicated")
        seen.add(edge)
        groups[a].append((index, line, keypoints[names[a]] * keypoints[names[b]]))
    buckets = [[], []]
    loads = [0, 0]
    for name, rows in sorted(groups.items(), key=lambda value: (-sum(row[2] for row in value[1]), value[0])):
        target = min(range(2), key=lambda index: (loads[index], index))
        buckets[target].extend(rows); loads[target] += sum(row[2] for row in rows)
    if not all(buckets):
        raise ValueError("The registered two-shard whole workload has an empty shard")
    lines = [[row[1] for row in sorted(bucket)] for bucket in buckets]
    if sorted(lines[0] + lines[1]) != sorted(pair_lines) or set(lines[0]) & set(lines[1]):
        raise AssertionError("The complete original pair partition changed")
    return lines, {"requested_pairs": len(pair_lines), "source_groups": len(groups), "shard_pairs": list(map(len, lines)),
                   "predicted_descriptor_product_loads_not_measured_GPU_cost": loads,
                   "within_shard_original_subsequence": True, "all_original_pairs_exactly_once": True}


def merge_databases(destination, source):
    with closing(readonly(destination)) as a, closing(readonly(source)) as b:
        if schemas(a) != schemas(b):
            raise ValueError("The native shard schemas differ before merge")
        names = {table: [row[1] for row in columns(a, table)] for table in ("matches", "two_view_geometries")}
        if any(columns(a, table) != columns(b, table) for table in names):
            raise ValueError("The native shard column declarations differ")
    inserted = {}
    with closing(sqlite3.connect(destination, uri=True)) as connection:
        connection.execute("ATTACH DATABASE ? AS owned_shard", ("file:" + Path(source).resolve().as_posix() + "?mode=ro",))
        try:
            connection.execute("BEGIN IMMEDIATE")
            for table, fields in names.items():
                q = quote(table)
                if connection.execute("SELECT 1 FROM main." + q + " a JOIN owned_shard." + q + " b USING(pair_id) LIMIT 1").fetchone():
                    raise ValueError("Native shard pair IDs overlap")
                values = ",".join(map(quote, fields))
                cursor = connection.execute("INSERT INTO main." + q + " (" + values + ") SELECT " + values + " FROM owned_shard." + q + " ORDER BY pair_id")
                inserted[table] = cursor.rowcount
                cursor.close()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.execute("DETACH DATABASE owned_shard")
    return {"inserted_rows": inserted, "no_original_or_shared_writer": True, "all_native_columns_preserved": True}


def gpu():
    values = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    total, used, free = map(int, values)
    if used > 12288 or free < 2048:
        raise RuntimeError("The registered whole-device GPU reserve was exceeded")
    return {"totalMiB": total, "usedMiB": used, "freeMiB": free, "scope": "whole_device_not_owned_peak"}


def stop_owned(records):
    for row in records:
        process = row["process"]
        if process.poll() is None:
            current = psutil.Process(process.pid)
            if current.create_time() != row["creation_time"]:
                raise RuntimeError("An owned native process birth changed")
            process.terminate()
    for row in records:
        process = row["process"]
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=10)


def run_native_jobs(folder, commands):
    records, logs, events = [], [], []
    started = time.perf_counter()
    next_gpu = started
    try:
        for index, command in enumerate(commands):
            log = (folder / ("worker" + str(index) + ".log")).open("x", encoding="utf-8")
            logs.append(log)
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            records.append({"process": process, "pid": process.pid, "creation_time": psutil.Process(process.pid).create_time(),
                            "command": command, "started_seconds": time.perf_counter() - started, "finished_seconds": None})
        while True:
            now = time.perf_counter()
            for row in records:
                result = row["process"].poll()
                if result is not None and row["finished_seconds"] is None:
                    row["finished_seconds"] = now - started
                if result is not None and result != 0:
                    raise ChildProcessError("The actual native matching process failed with exit " + str(result))
            if all(row["process"].poll() is not None for row in records):
                break
            if now - started > 120:
                raise RuntimeError("The registered native matching time bound was exceeded")
            if now >= next_gpu:
                events.append({"seconds": now - started, **gpu()}); next_gpu = now + 2
            time.sleep(.1)
    finally:
        stop_owned(records)
        for log in logs:
            log.close()
        save(folder / "NATIVE_PROCESSES.json", [{key: value for key, value in row.items() if key != "process"} | {"returncode": row["process"].returncode} for row in records])
        save(folder / "GPU_SAMPLES.json", events)
    intervals = [(row["started_seconds"], row["finished_seconds"]) for row in records]
    overlap = max(0., min(end for start, end in intervals) - max(start for start, end in intervals)) if len(intervals) == 2 else 0.
    return {"complete_native_jobs_seconds": time.perf_counter() - started, "native_processes": len(records),
            "actual_process_lifetime_overlap_seconds_not_GPU_kernel_concurrency": overlap, "all_native_processes_actually_waited": True}


def entry(folder, candidate, template, pair_lines, image_names):
    folder.mkdir()
    started = time.perf_counter()
    database = folder / "database.db"
    shutil.copyfile(OLD / "features.db", database)
    partition = merge = None
    paths = [(database, OLD / "guided_pairs.txt")]
    if candidate:
        lines, partition = partition_pairs(database, pair_lines)
        paths = []
        for index, subset in enumerate(lines):
            path = database if index == 0 else folder / "shard1.db"
            if index:
                shutil.copyfile(OLD / "features.db", path)
            pairs = folder / ("pairs" + str(index) + ".txt")
            pairs.write_text("\n".join(subset) + "\n", encoding="utf-8")
            paths.append((path, pairs))
    commands = []
    for path, pairs in paths:
        command = template.copy()
        command[command.index("--database_path") + 1] = str(path)
        command[command.index("--match_list_path") + 1] = str(pairs)
        commands.append(command)
    native = run_native_jobs(folder, commands)
    if candidate:
        merge = merge_databases(database, paths[1][0])
    summary = database_summary(database, image_names)
    atomic_json(folder / "MATCHING.json", summary)
    gc.collect()
    elapsed = time.perf_counter() - started
    result = {"candidate": candidate, "complete_closed_features_to_ready_database_seconds": elapsed,
              "database": str(database), "summary": summary, "partition": partition, "merge": merge, "native": native,
              "includes_copies_partition_all_startup_wait_merge_summary_publish_and_drain": True}
    save(folder / "ENTRY.json", result)
    print(json.dumps({"side": "sharded" if candidate else "native", "complete_seconds": elapsed,
                      "pairs": summary["attempted_matching_pairs"], "verified": summary["geometric_pairs_at_least_15_inliers"]}), flush=True)
    return result


def main():
    output = Path(sys.argv[1]).resolve()
    output.relative_to(OUT)
    order = sys.argv[2] if len(sys.argv) > 2 else "AB"
    if order not in ("AB", "BA") or output.exists() or shutil.disk_usage(str(ROOT)).free < 20 * 1024 ** 3:
        raise ValueError("A fresh registered private matching entry and order are required")
    output.mkdir(parents=True)
    stage = next(row for row in read(OLD / "STATE.json")["stages"] if row["name"] == "guided_matching")
    template = stage["command"].copy()
    if template[1] != "matches_importer" or template[template.index("--FeatureMatching.num_threads") + 1] != "8" or template[template.index("--FeatureMatching.gpu_index") + 1] != "0":
        raise ValueError("The original native matching interface changed")
    template[0] = str((ROOT / template[0]).resolve())
    sources = [data_identity(Path(__file__)), data_identity(V1 / "run_sfm_preflight.py"), data_identity(V1 / "runtime.py"), data_identity(Path(template[0]))]
    inputs = [data_identity(OLD / name) for name in ("features.db", "guided_pairs.txt", "image_list.txt", "STATE.json", "CONTRACT.json")]
    names = (OLD / "image_list.txt").read_text(encoding="utf-8").splitlines()
    pairs = (OLD / "guided_pairs.txt").read_text(encoding="utf-8").splitlines()
    if len(names) != 1002 or len(pairs) != 11583:
        raise AssertionError("The registered complete matching denominator changed")
    save(output / "PROTOCOL.json", {"sources": sources, "inputs": inputs, "actual_native_template": template, "order": order,
                                   "complete_images": 1002, "complete_requested_pairs": 11583, "viewport_or_Gaussian_render_not_this_stage": True,
                                   "max_output_bytes": 8 * 1024 ** 3, "registration_mapper_and_GT_not_run": True,
                                   "root_count": 35, "new_optimization_count": 0})
    (output / "sources").mkdir()
    for row in sources[:3]:
        shutil.copyfile(row["path"], output / "sources" / Path(row["path"]).name)
    before = gpu()
    started = time.perf_counter()
    entries = []
    try:
        for candidate in ((False, True) if order == "AB" else (True, False)):
            entries.append(entry(output / ("c" if candidate else "b"), candidate, template, pairs, names))
        b = next(row for row in entries if not row["candidate"])
        c = next(row for row in entries if row["candidate"])
        typed = compare_databases(b["database"], c["database"])
        base_features = compare_databases(OLD / "features.db", b["database"], exclude=("matches", "two_view_geometries"))
        candidate_features = compare_databases(OLD / "features.db", c["database"], exclude=("matches", "two_view_geometries"))
        checks = {"all_current_native_database_tables_columns_and_BLOB_values_exact": typed["complete_values_exact"],
                  "full_original_features_calibration_and_ID_tables_preserved": base_features["complete_values_exact"] and candidate_features["complete_values_exact"],
                  "original_full1002_and11583_summary_equal": b["summary"] == c["summary"] and b["summary"]["image_count"] == 1002 and b["summary"]["attempted_matching_pairs"] == 11583,
                  "actual_two_native_lifetimes_overlapped_and_all_children_waited": c["native"]["actual_process_lifetime_overlap_seconds_not_GPU_kernel_concurrency"] > 0 and all(row["native"]["all_native_processes_actually_waited"] for row in entries),
                  "all_original_pairs_partitioned_once_and_all_native_columns_merged": c["partition"]["all_original_pairs_exactly_once"] and c["merge"]["all_native_columns_preserved"],
                  "all_current_sources_and_inputs_unchanged": all(data_identity(row["path"]) == row for row in sources + inputs),
                  "registered_disk_and_total_candidate_output_bounds": shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3 and sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= 8 * 1024 ** 3}
        raw = (b["complete_closed_features_to_ready_database_seconds"] - c["complete_closed_features_to_ready_database_seconds"]) / b["complete_closed_features_to_ready_database_seconds"]
        previous = sum(read(OUT.parent / (name + "_GUARD.json"))["elapsed_seconds"] for name in ("p0", "p1", "keys"))
        known_seconds = time.perf_counter() - started + previous + 15
        upper = (b["complete_closed_features_to_ready_database_seconds"] - c["complete_closed_features_to_ready_database_seconds"]) / (b["complete_closed_features_to_ready_database_seconds"] + known_seconds / 6)
        save(output / "RESULT.json", {"status": "complete_fresh_matching_pair_exact_pending_formal0", "checks": checks, "entries": entries,
                                      "typed_complete_oracle": typed, "original_features_oracles": [base_features, candidate_features],
                                      "gpu_before": before, "gpu_after": gpu(), "raw_complete_stage_reduction": raw,
                                      "known_fee_upper_speed_not_acceptance": upper, "known_execution_fee_seconds_lower_bound_with_registered15_allowance": known_seconds,
                                      "source_reader_and_remaining_generic_formal_public_recovery_review_fees_not_yet_in_upper": True,
                                      "sources": sources, "inputs": inputs, "root_count": 35, "new_optimization_count": 0})
        print(json.dumps({"checks": checks, "raw_reduction": raw, "known_fee_speed_upper": upper, "new_count": 0}), flush=True)
        if not all(checks.values()):
            raise AssertionError("The complete native matching consumer failed verification")
    except BaseException as error:
        save(output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(), "entries_completed": entries,
                                       "sources": sources, "inputs": inputs, "root_count": 35, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
