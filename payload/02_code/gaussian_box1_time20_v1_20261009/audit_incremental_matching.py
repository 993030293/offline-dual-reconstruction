"""Repeat sparse native matching from identical complete features in private databases."""

import argparse
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
from atomic_json_windows import atomic_json
from gpu_lease import gpu_lease
from runtime import data_identity
from sqlite_verified_backup import verified_private_backup


def complete_native_tables_equal(first, second):
    result = []
    with closing(sqlite3.connect(first.as_uri()+"?mode=ro", uri=True)) as a, \
            closing(sqlite3.connect(second.as_uri()+"?mode=ro", uri=True)) as b:
        schema_query = "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"
        schema = a.execute(schema_query).fetchall()
        if schema != b.execute(schema_query).fetchall():
            raise AssertionError("The complete native table schemas differ")
        for name, _ in schema:
            escaped = '"'+name.replace('"', '""')+'"'
            columns = a.execute("PRAGMA table_info("+escaped+")").fetchall()
            if columns != b.execute("PRAGMA table_info("+escaped+")").fetchall():
                raise AssertionError("The complete native column definitions differ")
            keys = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
            order = ','.join('"'+key.replace('"', '""')+'"' for key in keys) if keys else 'rowid'
            query = "SELECT * FROM "+escaped+" ORDER BY "+order
            left, right = a.execute(query), b.execute(query)
            count = 0
            while True:
                one, two = left.fetchmany(32), right.fetchmany(32)
                if one != two:
                    raise AssertionError("An entire native row or blob differs: "+name)
                count += len(one)
                if not one:
                    break
            result.append({"table": name, "complete_rows": count, "all_values_and_blobs_equal": True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.augmentation / "database.db"
    pair_file = args.augmentation / "new_pairs.txt"
    master = args.output / "complete_source.db"
    backup_started = time.monotonic()
    backup = verified_private_backup(database, master, args.output / "source_bundle", data_identity)
    backup_seconds = time.monotonic()-backup_started
    signatures = [*backup["source_members"], data_identity(pair_file)]
    with closing(sqlite3.connect(master.as_uri()+"?mode=ro", uri=True)) as source:
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("The complete source native database is corrupt")
        names = {name: image_id for image_id, name in source.execute('SELECT image_id,name FROM images')}
        proposals = [tuple(line.split()) for line in pair_file.read_text().splitlines()]
        pair_ids = [2147483647*min(names[a],names[b])+max(names[a],names[b]) for a,b in proposals]
        expected = {table: {identity: source.execute('SELECT * FROM '+table+' WHERE pair_id=?',(identity,)).fetchone()
                           for identity in pair_ids} for table in ['matches','two_view_geometries']}
        if any(value is None for records in expected.values() for value in records.values()):
            raise ValueError("Every proposed native pair must have an original matching record")
    binary = ROOT / "05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    atomic_json(args.output / "CONTRACT.json", {"source": signatures, "binary": data_identity(binary),
        "complete_source_backup": backup, "source_backup_seconds": backup_seconds,
        "repeat": args.repeat, "all_descriptors_and_old_pair_records_retained": True,
        "new_pairs_removed_only_in_private_copies": True, "native_parameters_unchanged": True,
        "formal_reference_dense_matching_timing_is_reported_separately": True})
    results = []
    for index in range(args.repeat):
        folder = args.output / str(index)
        folder.mkdir()
        target = folder / "database.db"
        begin = time.monotonic()
        shutil.copyfile(master, target)
        with closing(sqlite3.connect(target)) as connection:
            with connection:
                for table in expected:
                    connection.executemany('DELETE FROM '+table+' WHERE pair_id=?',[(identity,) for identity in pair_ids])
        prepared = time.monotonic()-begin
        command = [binary, 'matches_importer', '--database_path', target, '--match_list_path', pair_file,
                   '--match_type','pairs','--FeatureMatching.use_gpu','1','--FeatureMatching.gpu_index','0',
                   '--FeatureMatching.num_threads','8','--TwoViewGeometry.random_seed','0','--default_random_seed','0']
        with gpu_lease(folder / "GPU_LEASE.json", 'incremental_repeat_'+str(index)):
            start = time.monotonic()
            with (folder / "matching.log").open('x') as stream:
                done = subprocess.run(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,timeout=180)
            seconds = time.monotonic()-start
        if done.returncode:
            raise RuntimeError("Native matching repeat failed")
        with closing(sqlite3.connect(target)) as connection:
            for table, records in expected.items():
                for identity, original in records.items():
                    if connection.execute('SELECT * FROM '+table+' WHERE pair_id=?',(identity,)).fetchone() != original:
                        raise AssertionError('An entire native pair row/blob differs: '+table+':'+str(identity))
        audit_started = time.monotonic()
        tables = complete_native_tables_equal(master, target)
        results.append({"repeat":index,"pair_count":len(pair_ids),"prepare_seconds":prepared,
                        "native_matching_seconds":seconds,"all_requested_native_pair_rows_and_blobs_exact":True,
                        "complete_native_tables": tables, "complete_audit_seconds": time.monotonic()-audit_started})
        atomic_json(args.output / "STATE.json", {"status":"running","results":results})
    if signatures != [data_identity(Path(record["path"])) for record in signatures]:
        raise ValueError("An immutable source changed during private matching repeats")
    atomic_json(args.output / "RESULT.json", {"status":"all_sparse_native_repeats_exact", "results":results,
        "source_unchanged":True,"reference_complete_run_not_replaced_by_this_component_test":True})
    print(results)


if __name__ == '__main__':
    main()
