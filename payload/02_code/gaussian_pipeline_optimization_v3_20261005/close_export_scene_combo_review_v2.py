"""Resolve equivalent timestamp encodings without repeating verified byte reads."""

import datetime
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(CODE), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]
from runtime import data_identity, atomic_json
from reaudit_export_scene_combo_v2 import read, OUT, DOC
from array_match_node import remaining


def main():
    folder = OUT / "v2r1"
    folder.mkdir(exist_ok=False)
    first = read(OUT / "v2r0/REVIEW.json")
    cost = read(DOC / "FULL_COST_REVISION_001.json")
    contract = read(OUT / "freeze_001_work/CONTRACT.json")
    bad = {key for key, value in first["checks"].items() if not value}
    expected_bad = {row["name"] + "_actual_old_fee_receipt_unchanged" for row in cost["receipt_inputs"]}
    checks = {"first_failure_only_timestamp_text_comparison": bad == expected_bad,
        "first_actual_full_outputs_and1086_input_review_retained": all(value for key, value in first["checks"].items() if key not in bad),
        "first_review_source_preserved": data_identity(first["source"]["path"]) == first["source"],
        "full1086_source_input_bytes_still_current": all(data_identity(row["path"]) == row for row in contract["sources_and_inputs"])}
    for row in cost["receipt_inputs"]:
        actual, saved = read(row["path"]), row["receipt"]
        differing = {key for key in set(actual) | set(saved) if actual.get(key) != saved.get(key)}
        checks[row["name"] + "_all_fee_identity_status_fields_and_timestamp_instants_exact"] = (
            differing == {"created_utc", "lease_released_utc"}
            and all(datetime.datetime.fromisoformat(actual[key]) == datetime.datetime.fromisoformat(saved[key]) for key in differing))
    first_lease, first_guard = read(OUT / "V2_REVIEW_LEASE_001.json"), read(OUT / "V2_REVIEW_GUARD_001.json")
    identities = [first["owner"], *first_guard["owned_process_identities"]]
    for prefix in ("guard", "child"):
        identities.append({"pid": first_lease[prefix + "_pid"], "creation_time": first_lease[prefix + "_creation_time"]})
    checks["first_failed_review_all_actual_owners_closed_and_lease_released"] = (
        first_lease["returncode"] == 1 and first_guard["returncode"] == 1
        and bool(first_lease.get("lease_released_utc")) and not remaining(identities))
    result = {**first, "status": "V2_exact_combination_review_verified_final_cost_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "first_timestamp_representation_failure_preserved": str(OUT / "v2r0/REVIEW.json"),
        "first_failed_review_complete_child_seconds": first_lease["child_elapsed_seconds"],
        "timestamp_diagnostic_two_reported_tool_wall_seconds": .8,
        "timestamp_normalization_scope": ["created_utc", "lease_released_utc"],
        "normalization_preserves_timezone_aware_instant_and_all_other_fields": True,
        "review_seconds_inside_script": time.perf_counter() - STARTED,
        "owner": {"pid": __import__("os").getpid(), "creation_time": __import__("psutil").Process().create_time()},
        "source": data_identity(Path(__file__)), "new_mechanism_contribution": 0}
    atomic_json(folder / "REVIEW.json", result)
    print({key: result[key] for key in ("status", "checks", "review_seconds_inside_script")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("A timestamp instant or original byte proof has changed")


if __name__ == "__main__":
    main()
