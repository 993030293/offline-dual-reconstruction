"""Profile the complete original prior-database preparation before mapper launch."""

import argparse
import cProfile
import importlib.util
import json
import os
from pathlib import Path
import pstats
import shutil
import sqlite3
import sys
import time
import types

import psutil


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from sqlite_evidence import compare_database_content


class MapperBoundary(Exception):
    """Stop the original consumer at its concrete subprocess launch boundary."""


def load_original():
    path = V1 / "run_prior_mapper.py"
    spec = importlib.util.spec_from_file_location("private_prior_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005")
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise RuntimeError("The diagnostic must leave20GiB free")
    source_db = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001/guided/database.db"
    source_wal = Path(str(source_db) + "-wal")
    initial = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p1_raw_combined_001"
    preflight = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_prior_mapper64_002/ACTIVATION_AUDIT.json"
    source_files = [Path(__file__), V1 / "run_prior_mapper.py", V1 / "runtime.py", V2 / "sqlite_evidence.py",
                    initial / "INITIAL_FRAME_MANIFEST.json", preflight, source_db]
    if source_wal.exists():
        source_files.append(source_wal)
    identities = [data_identity(path) for path in source_files]
    contract = {"mechanism": "R146_PRIOR_DATABASE_TRANSACTION", "diagnostic_original_only": True,
                "scope": "complete_original_database_preparation_before_mapper_launch",
                "pid": os.getpid(), "creation_time": psutil.Process().create_time(),
                "sources_and_inputs": identities, "expected_image_count": 1002,
                "candidate_implemented": False, "mapper_launched": False, "new_optimization_count": 0,
                "maximum_output_bytes": 30 * 1024**3, "quality_gate_relaxed": False}
    atomic_json(output / "CONTRACT.json", contract)
    snapshots = output / "executed_sources"
    snapshots.mkdir()
    for index, path in enumerate(source_files[:4]):
        shutil.copyfile(path, snapshots / (str(index) + "_" + path.name))

    # Both physical copies remain private; the source DB is never opened for writing.
    clone_dir = output / "source_with_wal"
    clone_dir.mkdir()
    clone = clone_dir / "database.db"
    db_only = output / "source_db_only.db"
    shutil.copyfile(source_db, clone)
    shutil.copyfile(source_db, db_only)
    if source_wal.exists():
        shutil.copyfile(source_wal, Path(str(clone) + "-wal"))
    for path, identity in zip(source_files, identities):
        if data_identity(path) != identity:
            raise ValueError("A source changed during the private snapshot copy")
    prepared_source = output / "closed_complete_source.db"
    with sqlite3.connect(clone) as connection, sqlite3.connect(prepared_source) as target:
        connection.backup(target)
    connection.close()
    target.close()
    wal_comparison = {"all_schema_and_rows_and_blobs_exact": False}
    try:
        wal_comparison = compare_database_content(db_only, prepared_source)
    except ValueError as error:
        # This explicitly recorded diagnostic difference is not a successful equivalence check.
        wal_comparison = {"all_schema_and_rows_and_blobs_exact": False,
                          "difference": str(error), "classification": "failed_verification_db_only_copy"}
    with sqlite3.connect("file:" + prepared_source.as_posix() + "?mode=ro", uri=True) as connection:
        counts = {table: connection.execute('SELECT count(*) FROM "' + table + '"').fetchone()[0]
                  for table in ("images", "keypoints", "descriptors", "matches", "two_view_geometries", "pose_priors")}
    connection.close()
    if counts["images"] != 1002 or counts["pose_priors"] != 0:
        raise ValueError("The actual full source is not the expected prior-free1002 database")
    atomic_json(output / "INPUT_SNAPSHOT.json", {"source_before_after_exact": True, "counts": counts,
                "source_with_WAL": data_identity(prepared_source), "db_only_vs_WAL_complete": wal_comparison,
                "snapshot_complete_seconds": time.perf_counter() - started, "new_optimization_count": 0})

    module = load_original()
    intercepted = []

    def stop_mapper(command, **kwargs):
        if len(command) < 2 or command[1] != "pose_prior_mapper":
            raise AssertionError("The concrete original mapper boundary changed")
        intercepted.append(list(command))
        raise MapperBoundary("Original database preparation completed; mapper deliberately not launched")

    # Bind a private module object without patching shared subprocess or the source file.
    module.subprocess = types.SimpleNamespace(Popen=stop_mapper, STDOUT=module.subprocess.STDOUT)
    prior_argv = sys.argv
    sys.argv = [str(V1 / "run_prior_mapper.py"), "--database", str(prepared_source),
                "--initial", str(initial), "--output", str(output / "original_preparation"),
                "--preflight-receipt", str(preflight), "--threads", "8", "--maximum-seconds", "1200"]
    profiler = cProfile.Profile()
    stage_started = time.perf_counter()
    boundary_reached = False
    try:
        profiler.enable()
        module.main()
    except MapperBoundary:
        boundary_reached = True
    finally:
        profiler.disable()
        sys.argv = prior_argv
    stage_seconds = time.perf_counter() - stage_started
    if not boundary_reached or len(intercepted) != 1:
        raise AssertionError("The original full preparation did not reach exactly one mapper launch")
    profiler.dump_stats(output / "original_preparation.prof")
    rows = []
    writes = []
    for (filename, line, name), (primitive_calls, calls, own_seconds, cumulative_seconds, _) in pstats.Stats(profiler).stats.items():
        row = {"file": filename, "line": line, "name": name, "calls": calls,
               "own_seconds": own_seconds, "cumulative_seconds": cumulative_seconds}
        rows.append(row)
        if "write_pose_prior" in name:
            writes.append(row)
    if not writes or sum(row["calls"] for row in writes) != 1002:
        raise AssertionError("The profiler did not observe all1002 actual native prior writes")
    write_seconds = sum(row["own_seconds"] for row in writes)
    original_contract = json.loads((output / "original_preparation/CONTRACT.json").read_text())
    if len(original_contract["prior_bindings"]) != 1002 or not original_contract["readback_verified"]:
        raise AssertionError("The original complete prior read-back is missing")
    prepared_counts = {}
    with sqlite3.connect("file:" + (output / "original_preparation/database.db").as_posix() + "?mode=ro", uri=True) as connection:
        prepared_counts = {table: connection.execute('SELECT count(*) FROM "' + table + '"').fetchone()[0]
                           for table in counts}
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise AssertionError("Prepared prior database integrity did not pass")
    connection.close()
    if prepared_counts != {**counts, "pose_priors": 1002}:
        raise AssertionError("The original full preparation changed another table denominator")
    final_sources = [data_identity(path) for path in source_files]
    if final_sources != identities:
        raise ValueError("The diagnostic changed an original source or input")
    result = {"status": "original_complete_preparation_profiled_candidate_not_implemented",
              "complete_original_preparation_seconds": stage_seconds,
              "native_prior_write_seconds": write_seconds,
              "native_prior_write_fraction_of_complete_phase": write_seconds / stage_seconds,
              "actual_write_calls": sum(row["calls"] for row in writes), "write_profile": writes,
              "profile_top_by_cumulative": sorted(rows, key=lambda row: -row["cumulative_seconds"])[:18],
              "counts_before": counts, "counts_after": prepared_counts,
              "original_all1002_binding_position_covariance_readback_verified": True,
              "all_original_sources_and_inputs_unchanged": True,
              "DB_only_copy_vs_WAL_complete": wal_comparison,
              "mapper_launches": 0, "candidate_implemented": False,
              "new_optimization_count": 0, "complete_diagnostic_seconds": time.perf_counter() - started,
              "profile_is_diagnostic_not_formal_three_pairs": True}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in (
        "status", "complete_original_preparation_seconds", "native_prior_write_seconds",
        "native_prior_write_fraction_of_complete_phase", "complete_diagnostic_seconds",
        "DB_only_copy_vs_WAL_complete", "new_optimization_count")}), flush=True)


if __name__ == "__main__":
    main()
