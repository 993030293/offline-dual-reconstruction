"""Check native shard contracts, a full public database, errors, and actual SIGINT."""

from contextlib import closing
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import threading
import time
import traceback

import psutil
import pycolmap

import native_sfm_match_shards as native

ROOT = native.ROOT
OUT = native.OUT
PUBLIC = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/database.db"


def closed(row):
    try:
        return abs(psutil.Process(row["pid"]).create_time() - row["creation_time"]) > .01
    except psutil.NoSuchProcess:
        return True


def output_bound():
    total = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    if total > 8 * 1024 ** 3 or shutil.disk_usage(str(ROOT)).free < 20 * 1024 ** 3:
        raise RuntimeError("The registered candidate disk bound was exceeded")
    return total


def must_raise(operation, kinds):
    try:
        operation()
    except kinds as error:
        return {"kind": type(error).__name__, "message": str(error)}
    raise AssertionError("A malformed native shard input was silently accepted")


def fixture(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            "CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT UNIQUE);"
            "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);"
            "CREATE TABLE matches(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);"
            "CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB,"
            "config INTEGER,F BLOB,E BLOB,H BLOB,qvec BLOB,tvec BLOB,extra_real REAL,extra_text TEXT);"
            "CREATE TABLE preserved(id INTEGER PRIMARY KEY,payload BLOB,label TEXT);"
        )
        connection.executemany("INSERT INTO images VALUES(?,?)", [(11, "a"), (37, "b"), (51, "c"), (93, "d")])
        connection.executemany("INSERT INTO keypoints VALUES(?,?,?,?)", [(11, 0, 4, b""), (37, 7, 4, b"b"), (51, 1001, 4, b"c"), (93, 13, 4, b"d")])
        connection.execute("INSERT INTO preserved VALUES(1,?,?)", (b"\x00\xff\x80", "arbitrary object"))
        connection.commit()


def generic(folder):
    folder.mkdir()
    source = folder / "template.db"
    fixture(source)
    pairs = ["a b", "a c", "b c", "b d", "c d", "d a"]
    first, trace = native.partition_pairs(source, pairs)
    second, repeat = native.partition_pairs(source, pairs)
    rejections = []
    for invalid in (["a a"], ["a unknown"], ["a b", "b a"], ["a b", "a b"], [], ["a b"], ["a"], ["a b c"]):
        rejections.append(must_raise(lambda value=invalid: native.partition_pairs(source, value), (ValueError, KeyError)))
    expected, left, right = [folder / name for name in ("expected.db", "left.db", "right.db")]
    for path in (expected, left, right):
        shutil.copyfile(source, path)
    for path, ids in ((expected, (7, 29)), (left, (7,)), (right, (29,))):
        with closing(sqlite3.connect(path)) as connection:
            for pair_id in ids:
                connection.execute("INSERT INTO matches VALUES(?,?,?,?)", (pair_id, 3, 2, b"\x00\xff\x81" + bytes([pair_id])))
                connection.execute("INSERT INTO two_view_geometries VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pair_id, 3, 2, b"\x00\x80", pair_id % 3, b"F", None, b"H\x00", b"q", b"t", .123456789, "texture independent"))
            connection.commit()
    merged = native.merge_databases(left, right)
    exact = native.compare_databases(expected, left)
    before = folder / "before_conflict.db"
    shutil.copyfile(left, before)
    overlap = must_raise(lambda: native.merge_databases(left, right), (ValueError,))
    overlap_unchanged = native.compare_databases(before, left)
    rollback_left, rollback_right = folder / "rollback_left.db", folder / "rollback_right.db"
    shutil.copyfile(source, rollback_left); shutil.copyfile(source, rollback_right)
    with closing(sqlite3.connect(rollback_left)) as connection:
        connection.execute("INSERT INTO two_view_geometries(pair_id,rows) VALUES(29,3)"); connection.commit()
    with closing(sqlite3.connect(rollback_right)) as connection:
        connection.execute("INSERT INTO matches VALUES(29,3,2,?)", (b"must rollback",))
        connection.execute("INSERT INTO two_view_geometries(pair_id,rows) VALUES(29,3)"); connection.commit()
    rollback_before = folder / "rollback_before.db"
    shutil.copyfile(rollback_left, rollback_before)
    second_table_conflict = must_raise(lambda: native.merge_databases(rollback_left, rollback_right), (ValueError,))
    rollback_exact = native.compare_databases(rollback_before, rollback_left)
    mismatch = folder / "mismatch.db"
    shutil.copyfile(right, mismatch)
    with closing(sqlite3.connect(mismatch)) as connection:
        connection.execute("ALTER TABLE matches ADD COLUMN unexpected INTEGER"); connection.commit()
    schema_failure = must_raise(lambda: native.merge_databases(left, mismatch), (ValueError,))
    unknown = folder / "unsupported.db"
    with closing(sqlite3.connect(unknown)) as connection:
        connection.execute("CREATE TABLE unknown(value INTEGER)"); connection.commit()
    unsupported_failure = must_raise(lambda: native.merge_databases(left, unknown), (ValueError,))
    checks = {
        "mixed_zero_and_large_feature_counts_nonsequential_IDs_deterministic": first == second and trace == repeat,
        "all_original_edges_once_source_groups_and_subsequence_preserved": sorted(sum(first, [])) == sorted(pairs) and all([pairs.index(line) for line in lines] == sorted(pairs.index(line) for line in lines) for lines in first),
        "eight_invalid_pair_population_cases_rejected": len(rejections) == 8,
        "all_generic_columns_NULLs_BLOBs_and_extra_values_exact": exact["complete_values_exact"] and merged["all_native_columns_preserved"],
        "duplicate_pair_failure_preserves_original_destination": overlap_unchanged["complete_values_exact"],
        "second_table_conflict_rolls_back_first_table_insertion": rollback_exact["complete_values_exact"],
        "mismatched_and_unsupported_schema_fail_without_fallback": bool(schema_failure and unsupported_failure),
    }
    result = {"checks": checks, "partition": trace, "exact": exact, "rejections": rejections,
              "overlap": overlap, "second_table_conflict": second_table_conflict,
              "schema_failure": schema_failure, "unsupported_failure": unsupported_failure}
    native.save(folder / "RESULT.json", result)
    if not all(checks.values()):
        raise AssertionError("The generic native shard contract failed")
    return result


def keep_pairs(path, pair_ids):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TEMP TABLE selected(pair_id INTEGER PRIMARY KEY)")
        connection.executemany("INSERT INTO selected VALUES(?)", [(value,) for value in pair_ids])
        for table in ("matches", "two_view_geometries"):
            connection.execute("DELETE FROM " + native.quote(table) + " WHERE pair_id NOT IN (SELECT pair_id FROM selected)")
        connection.commit()


def public_component(folder):
    folder.mkdir()
    with closing(native.readonly(PUBLIC)) as connection:
        names = dict(connection.execute("SELECT image_id,name FROM images").fetchall())
        pair_ids = [row[0] for row in connection.execute("SELECT pair_id FROM matches ORDER BY pair_id")]
        verified_ids = [row[0] for row in connection.execute("SELECT pair_id FROM two_view_geometries ORDER BY pair_id")]
    if len(names) != 360 or len(pair_ids) != 64620 or pair_ids != verified_ids:
        raise AssertionError("The original full public development database denominator changed")
    pairs, pair_lookup = [], {}
    for pair_id in pair_ids:
        a, b = pycolmap.pair_id_to_image_pair(pair_id)
        if pycolmap.image_pair_to_pair_id(a, b) != pair_id:
            raise AssertionError("The actual installed native pair-ID conversion failed")
        line = names[a] + " " + names[b]
        pairs.append(line); pair_lookup[line] = pair_id
    lines, trace = native.partition_pairs(PUBLIC, pairs)
    left, right = folder / "left.db", folder / "right.db"
    for index, path in enumerate((left, right)):
        shutil.copyfile(PUBLIC, path)
        keep_pairs(path, [pair_lookup[line] for line in lines[index]])
    merge = native.merge_databases(left, right)
    exact = native.compare_databases(PUBLIC, left)
    preserved = native.compare_databases(PUBLIC, right, exclude=("matches", "two_view_geometries"))
    checks = {"full_original_public360_all64620_pairs_preserved": trace["requested_pairs"] == 64620 and exact["counts"]["images"] == 360 and exact["counts"]["matches"] == exact["counts"]["two_view_geometries"] == 64620,
              "every_public_schema_column_raw_value_and_BLOB_exact": exact["complete_values_exact"],
              "public_feature_and_calibration_ID_tables_preserved": preserved["complete_values_exact"],
              "actual_original_pair_ID_api_roundtrip_and_partition_merge": trace["all_original_pairs_exactly_once"] and merge["all_native_columns_preserved"]}
    result = {"checks": checks, "exact": exact, "partition": trace, "merge": merge,
              "already_observed_public_development_component_not_blind_GT_or_new_native_match": True,
              "no_public_mapper_or_Gaussian_training_or_new_object_quality_claim": True}
    native.save(folder / "RESULT.json", result)
    if not all(checks.values()):
        raise AssertionError("The full public native shard component failed")
    return result


def lifecycle(folder, template, candidate, cancel):
    folder.mkdir()
    commands = []
    count = 2 if candidate else 1
    pair_lines = (native.OLD / "guided_pairs.txt").read_text(encoding="utf-8").splitlines()
    subsets = native.partition_pairs(native.OLD / "features.db", pair_lines)[0] if candidate else [pair_lines]
    for index in range(count):
        database = folder / ("owned" + str(index) + ".db")
        pairs = folder / ("pairs" + str(index) + ".txt")
        shutil.copyfile(native.OLD / "features.db", database)
        pairs.write_text("\n".join(subsets[index]) + "\n", encoding="utf-8")
        command = template.copy()
        command[command.index("--database_path") + 1] = str(database)
        command[command.index("--match_list_path") + 1] = str(pairs)
        if not cancel and index == count - 1:
            command.extend(["--R165_deliberately_unknown_native_contract_flag", "1"])
        commands.append(command)
    owned = []
    captured = None
    original = native.subprocess.Popen
    active = threading.Event(); active.set()
    evidence = {"sent": False, "matching_workers_live_at_signal": []}

    def record_popen(command, *args, **kwargs):
        process = original(command, *args, **kwargs)
        if isinstance(command, list) and len(command) > 1 and command[1] == "matches_importer":
            owned.append({"process": process, "pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()})
        return process

    def observe_and_signal():
        deadline = time.perf_counter() + 15
        while active.is_set() and time.perf_counter() < deadline:
            rows = list(owned)
            ready = len(rows) == count and all(row["process"].poll() is None for row in rows)
            logs = [folder / ("worker" + str(index) + ".log") for index in range(count)]
            ready = ready and all(path.exists() and "Processing block [1/" in path.read_text(encoding="utf-8", errors="replace") for path in logs)
            if ready and active.is_set():
                births = []
                for row in rows:
                    births.append({key: value for key, value in row.items() if key != "process"})
                    for child in psutil.Process(row["pid"]).children(recursive=True):
                        births.append({"pid": child.pid, "creation_time": child.create_time()})
                evidence.update(sent=True, matching_workers_live_at_signal=births,
                    original_complete1002_matching_native_block_active=True,
                    signal_number=int(signal.SIGINT), signal_sender="owned_observer_thread_C_raise_SIGINT_to_current_Python_process")
                native.save(folder / "ACTUAL_SIGNAL.json", evidence)
                signal.raise_signal(signal.SIGINT)
                return
            time.sleep(.005)

    native.subprocess.Popen = record_popen
    observer = threading.Thread(target=observe_and_signal, name="owned_native_matching_SIGINT") if cancel else None
    if observer:
        observer.start()
    try:
        native.run_native_jobs(folder, commands)
    except (ChildProcessError, KeyboardInterrupt) as error:
        captured = {"kind": type(error).__name__, "message": str(error)}
    finally:
        active.clear()
        if observer:
            observer.join(timeout=20)
        native.subprocess.Popen = original
    if observer and observer.is_alive():
        raise AssertionError("The actual cancellation observer did not drain")
    native_rows = native.read(folder / "NATIVE_PROCESSES.json")
    births = [{key: value for key, value in row.items() if key != "process"} for row in owned] + evidence["matching_workers_live_at_signal"]
    deadline = time.perf_counter() + 5
    while not all(closed(row) for row in births) and time.perf_counter() < deadline:
        time.sleep(.01)
    checks = {"actual_expected_native_error_or_SIGINT_propagated": captured is not None and captured["kind"] == ("KeyboardInterrupt" if cancel else "ChildProcessError"),
              "all_owned_native_workers_waited_and_exact_births_closed": len(native_rows) == count and all(row["returncode"] is not None for row in native_rows) and all(closed(row) for row in births),
              "original_subprocess_binding_restored_and_observer_drained": native.subprocess.Popen is original and (observer is None or not observer.is_alive()),
              "no_success_fallback_or_ready_database_published": not (folder / "MATCHING.json").exists() and not (folder / "ENTRY.json").exists()}
    if cancel:
        checks["real_SIGINT_after_all_actual_native_matching_blocks_became_active"] = evidence["sent"] and evidence["original_complete1002_matching_native_block_active"]
    elif candidate:
        checks["one_real_native_error_terminates_and_waits_the_other_live_worker"] = native_rows[0]["finished_seconds"] is None and native_rows[0]["returncode"] != 0 and native_rows[1]["returncode"] != 0
    result = {"candidate": candidate, "cancel": cancel, "checks": checks, "captured": captured,
              "signal_evidence": evidence, "native_rows": native_rows, "closed_births": births,
              "partial_private_databases_kept_not_accepted": True}
    native.save(folder / "RESULT.json", result)
    if not all(checks.values()):
        raise AssertionError("The actual native shard lifecycle contract failed")
    return result


def main():
    started = time.perf_counter()
    output = OUT / "checks0"
    if output.exists():
        raise FileExistsError("A fresh native matching contract directory is required")
    output.mkdir()
    sources = [native.data_identity(Path(__file__)), native.data_identity(Path(native.__file__))]
    inputs = [native.data_identity(PUBLIC), native.data_identity(native.OLD / "features.db"),
              native.data_identity(native.OLD / "guided_pairs.txt"), native.data_identity(native.OLD / "STATE.json")]
    (output / "sources").mkdir()
    for row in sources:
        shutil.copyfile(row["path"], output / "sources" / Path(row["path"]).name)
    template = next(row["command"].copy() for row in native.read(native.OLD / "STATE.json")["stages"] if row["name"] == "guided_matching")
    template[0] = str((ROOT / template[0]).resolve())
    native.save(output / "PROTOCOL.json", {"sources": sources, "inputs": inputs,
        "task": str(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/R165_NATIVE_MATCH_SHARDS_TASK.md"),
        "frozen_matching_algorithm_not_modified_by_validation": True, "new_optimization_count": 0})
    results = {}
    try:
        results["generic"] = generic(output / "generic")
        results["public_component"] = public_component(output / "public")
        output_bound()
        for name, candidate, cancel in (("native_error", False, False), ("shard_error", True, False),
                                        ("native_cancel", False, True), ("shard_cancel", True, True)):
            results[name] = lifecycle(output / name, template, candidate, cancel)
            output_bound()
        checks = {"all_complete_generic_public_and_native_lifecycle_checks_passed": all(all(row["checks"].values()) for row in results.values()),
                  "all_scientific_sources_and_input_databases_unchanged": all(native.data_identity(row["path"]) == row for row in sources + inputs),
                  "registered_candidate_disk_and_GPU_bounds": bool(output_bound() and native.gpu())}
        native.save(output / "RESULT.json", {"status": "native_shard_contracts_passed_recovery_and_root_review_pending0", "checks": checks,
            "components": results, "sources": sources, "inputs": inputs, "elapsed_seconds": time.perf_counter() - started,
            "root_count": 35, "new_optimization_count": 0})
        print(json.dumps({"checks": checks, "components": {name: row["checks"] for name, row in results.items()}, "new_count": 0}), flush=True)
        if not all(checks.values()):
            raise AssertionError("The complete native shard acceptance supplements failed")
    except BaseException as error:
        native.save(output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(), "completed": results,
            "sources": sources, "inputs": inputs, "elapsed_seconds": time.perf_counter() - started, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
