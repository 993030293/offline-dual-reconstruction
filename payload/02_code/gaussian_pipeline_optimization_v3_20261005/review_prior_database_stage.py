"""Review the original full prior-preparation diagnostic and its closed resources."""

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def same_other_tables(source, prepared):
    connections = [sqlite3.connect("file:" + path.as_posix() + "?mode=ro", uri=True)
                   for path in (source, prepared)]
    try:
        for connection in connections:
            connection.execute("BEGIN")
        schemas = [connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
                   for connection in connections]
        if schemas[0] != schemas[1]:
            raise AssertionError("Original preparation changed the complete database schema")
        tables = []
        for kind, name, _, _ in schemas[0]:
            if kind != "table" or name in ("pose_priors", "sqlite_sequence"):
                continue
            quoted = '"' + name.replace('"', '""') + '"'
            fields = connections[0].execute("PRAGMA table_info(" + quoted + ")").fetchall()
            primary = sorted((row[5], row[1]) for row in fields if row[5])
            order = [column for _, column in primary] if primary else [row[1] for row in fields]
            query = "SELECT * FROM " + quoted + " ORDER BY " + ",".join('"' + column.replace('"', '""') + '"' for column in order)
            cursors = [connection.execute(query) for connection in connections]
            count, blob_bytes = 0, 0
            while True:
                first, second = (cursor.fetchone() for cursor in cursors)
                if first != second:
                    raise AssertionError("Original preparation changed input content in " + name)
                if first is None:
                    break
                count += 1
                blob_bytes += sum(len(value) for value in first if isinstance(value, bytes))
            tables.append({"name": name, "rows": count, "blob_bytes_compared": blob_bytes})
        sequences = [dict(connection.execute("SELECT name,seq FROM sqlite_sequence")) for connection in connections]
        if sequences[1] != sequences[0]:
            raise AssertionError("The original preparation changed another ID sequence")
        if connections[1].execute("SELECT count(*) FROM pose_priors").fetchone()[0] != 1002:
            raise AssertionError("The original full prior denominator changed")
        return tables
    finally:
        for connection in connections:
            connection.rollback()
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "source_snapshot.py")
    own = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pt001"
    checks = {}
    result = json.loads((own / "d1/RESULT.json").read_text())
    first_log = (own / "d0.log").read_text(encoding="utf-8")
    checks["first_real_STDOUT_binding_failure_preserved"] = "AttributeError" in first_log and "STDOUT" in first_log
    first_source = (own / "d0/executed_sources/0_diagnose_prior_database_stage.py").read_text()
    current_source = (own / "d1/executed_sources/0_diagnose_prior_database_stage.py").read_text()
    expected = first_source.replace("types.SimpleNamespace(Popen=stop_mapper)",
                                    "types.SimpleNamespace(Popen=stop_mapper, STDOUT=module.subprocess.STDOUT)")
    checks["only_private_instrumentation_STDOUT_constant_fixed"] = expected == current_source and first_source != current_source
    checks["actual_all1002_native_write_calls_profiled"] = result["actual_write_calls"] == 1002
    checks["complete_phase_profile_is_not_a_formal_speed_pair"] = result["profile_is_diagnostic_not_formal_three_pairs"]
    checks["native_write_share_below_pre_registered5percent"] = 0 < result["native_prior_write_fraction_of_complete_phase"] < .05
    checks["candidate_not_implemented_or_accepted"] = not result["candidate_implemented"] and result["new_optimization_count"] == 0
    checks["no_mapper_launched"] = result["mapper_launches"] == 0 and not (own / "d1/original_preparation/STATE.json").exists()
    checks["all_original_readback_and_source_guards_passed"] = (
        result["original_all1002_binding_position_covariance_readback_verified"]
        and result["all_original_sources_and_inputs_unchanged"])
    checks["DB_only_and_WAL_complete_all_schema_rows_blobs_exact"] = result["DB_only_copy_vs_WAL_complete"]["all_schema_and_rows_and_blobs_exact"]
    identities = []
    snapshot_count = 0
    closed = []
    costs = 0.
    for label in ("d0", "d1"):
        contract = json.loads((own / label / "CONTRACT.json").read_text())
        for record in contract["sources_and_inputs"]:
            if record["path"] == str(ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005/diagnose_prior_database_stage.py"):
                continue
            if data_identity(record["path"]) != record:
                raise AssertionError("A protected current source or input changed")
            identities.append(record)
        for index, record in enumerate(contract["sources_and_inputs"][:4]):
            saved = own / label / "executed_sources" / (str(index) + "_" + Path(record["path"]).name)
            if data_identity(saved)["sha256"] != record["sha256"]:
                raise AssertionError("The actually executed source snapshot changed")
            snapshot_count += 1
        process = json.loads((own / (label + "_process.json")).read_text())
        lease = json.loads((own / (label + "_lease.json")).read_text())
        expected_code = 1 if label == "d0" else 0
        if process["returncode"] != expected_code or lease["returncode"] != expected_code or not lease.get("lease_released_utc"):
            raise AssertionError("The actual failed or completed process was not drained and released")
        costs += lease["child_elapsed_seconds"]
        records = process["owned_process_identities"] + [
            {"pid": contract["pid"], "creation_time": contract["creation_time"]},
            {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
        for record in records:
            try:
                if psutil.Process(record["pid"]).create_time() == record["creation_time"]:
                    raise AssertionError("An owned diagnostic process remains live")
            except psutil.NoSuchProcess:
                pass
            closed.append(record)
    checks["all_executed_snapshot_bytes_exact"] = snapshot_count == 8
    checks["all_original_current_input_source_identities_exact"] = bool(identities)
    tables = same_other_tables(own / "d1/closed_complete_source.db", own / "d1/original_preparation/database.db")
    checks["all_other_schema_rows_blobs_and_ID_sequences_exact"] = len(tables) == 10
    checks["actual_original1002_images11583_matches_input_preserved"] = result["counts_before"]["images"] == 1002 and result["counts_before"]["matches"] == 11583
    checks["owned_preceding_processes_and_leases_closed"] = bool(closed)
    checks["output_and_disk_budget_preserved"] = sum(path.stat().st_size for path in own.rglob("*") if path.is_file()) < 30 * 1024**3 and shutil.disk_usage(own).free > 20 * 1024**3
    if not all(checks.values()):
        raise AssertionError("The separate root self-review did not pass all required diagnostic checks")
    report = {"status": "deferred_original_phase_native_write_share1point68percent_candidate_unimplemented_count0",
              "checks": checks, "review_kind": "separate_root_self_review_no_subagents",
              "input_records_checked": identities, "source_snapshots_checked": snapshot_count,
              "all_other_tables_exact": tables, "owned_exited_process_records": closed,
              "diagnostic_complete_lease_child_seconds": costs,
              "review_seconds": time.perf_counter() - started, "new_optimization_count": 0,
              "not_run": ["transaction_implementation", "generic_transaction_error_contract", "three_fresh_formal_pairs",
                          "actual_mapper_or_BA", "public_or_unseen_object", "new_training", "production_deployment"]}
    atomic_json(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "checks": len(checks), "known_diagnostic_seconds": costs}), flush=True)


if __name__ == "__main__":
    main()
