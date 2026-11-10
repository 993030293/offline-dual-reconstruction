"""Resolve integer-key JSON serialization without weakening the full summary oracle."""
import datetime
import json
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup009"
OLD = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from run_sfm_preflight import database_summary
from runtime import data_identity
from review_orbit_convergence_dag import closed


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    inputs = {}
    births = []
    fee = 0.
    for name in ("p0", "p1"):
        protocol = read(OUT / name / "PROTOCOL.json")
        guard, lease = read(OUT / (name + "_GUARD.json")), read(OUT / (name + "_lease.json"))
        fee += guard["elapsed_seconds"]
        births.extend(guard["owned_process_identities"])
        births.extend({"pid": lease[k + "_pid"], "creation_time": lease[k + "_creation_time"]} for k in ("guard", "child"))
        for row in protocol["inputs"] + protocol["sources"]:
            assert row["path"] not in inputs or inputs[row["path"]] == row
            inputs[row["path"]] = row
    names = (OLD / "image_list.txt").read_text(encoding="utf-8").splitlines()
    cases = []
    for database, saved, current in (("features.db", "FEATURES.json", "ORIGINAL_FEATURES.json"), ("guided/database.db", "guided/MATCHING.json", "ORIGINAL_MATCHING.json")):
        raw = database_summary(OLD / database, names)
        expected = read(OLD / saved)
        persisted = json.loads(json.dumps(raw, allow_nan=False))
        case = {"database": database, "raw_differing_fields": [key for key in expected if raw[key] != expected[key]],
                "raw_counter_key_types": sorted({type(key).__name__ for key in raw["geometric_configuration_counts"]}),
                "stored_counter_key_types": sorted({type(key).__name__ for key in expected["geometric_configuration_counts"]}),
                "whole_persisted_output_exact": persisted == expected == read(OUT / "p1" / current)}
        cases.append(case)
    checks = {
        "whole_original_current_and_historical_persisted_outputs_exact": all(row["whole_persisted_output_exact"] for row in cases),
        "only_typed_counter_key_boundary_not_actual_SQLite_drift": cases[0]["raw_differing_fields"] == [] and cases[1]["raw_differing_fields"] == ["geometric_configuration_counts"] and cases[1]["raw_counter_key_types"] == ["int"] and cases[1]["stored_counter_key_types"] == ["str"],
        "p0_failed_raw_dictionary_comparison_preserved": (OUT / "p0/FAILURE.json").exists() and read(OUT / "p0_GUARD.json")["returncode"] == 1,
        "p1_repeat_current_all6_checks_passed_no_candidate_acceptance": all(read(OUT / "p1/RESULT.json")["checks"].values()) and read(OUT / "p1_GUARD.json")["returncode"] == 0,
        "all_current_complete_inputs_and_code_unchanged": all(data_identity(row["path"]) == row for row in inputs.values()),
        "all_prior_birth_records_ended": all(closed(row) for row in births),
    }
    payload = {"status": "original_complete_summary_JSON_boundary_resolved_no_SQLite_drift_observed0" if all(checks.values()) else "failed_summary_key_boundary_verification0",
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": checks, "cases": cases,
               "source_inputs": list(inputs.values()) + [data_identity(Path(__file__))], "ended_process_birth_records": births,
               "reviewer_process_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
               "known_p0_p1_guard_seconds": fee, "p0_actual_tool": {"chunk": "e9ed72", "exit_code": 1},
               "p1_actual_tool": {"chunk": "a61527", "exit_code": 0}, "failed_shell_reader": {"chunk": "408970", "exit_code": 1, "seconds": .1068682},
               "parent_count": 35, "new_optimization_count": 0, "goal_status": "active",
               "old_source_mismatch_classification_superseded_by_JSON_key_evidence": True}
    with (OUT / "SOURCE_ORACLE_KEYS_001.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checks": checks, "cases": cases, "new_count": 0}))
    if not all(checks.values()):
        raise AssertionError("The full original JSON-boundary verification failed")


if __name__ == "__main__":
    main()
