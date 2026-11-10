"""Verify transient recovery, permanent denial and protected path rejection."""

import argparse
import ctypes
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from check_bounded_journal_task_002 import hold_read
from owned_json_write import OwnedAtomicJSON, owned_atomic_json, ORIGINAL
import runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001")
    started = time.perf_counter()
    sources = [Path(__file__), Path(__file__).with_name("owned_json_write.py")]
    identities = [data_identity(p) for p in sources]
    checks = {}
    operator = OwnedAtomicJSON(args.output)
    transient = args.output/"transient.json"
    ORIGINAL(transient, {"value": 0})
    handle, close = hold_read(transient)
    closer = threading.Thread(target=lambda: (time.sleep(.05), close(handle)))
    closer.start()
    operator(transient, {"value": 1})
    closer.join()
    checks["transient_exact_atomic_payload_and_recorded_retries"] = json.loads(transient.read_text()) == {"value": 1} and operator.trace["events"][0]["attempts"] > 1
    permanent = args.output/"permanent.json"
    ORIGINAL(permanent, {"value": 0})
    handle, close = hold_read(permanent)
    try:
        before = time.perf_counter()
        try:
            operator(permanent, {"value": 1})
            checks["permanent_same_permission_error_and_old_payload"] = False
        except PermissionError as error:
            checks["permanent_same_permission_error_and_old_payload"] = error.winerror == 5 and json.loads(permanent.read_text()) == {"value": 0}
        checks["permanent_retry_bounded_to_one_original_monitor_interval"] = .5 <= time.perf_counter()-before < 1.
    finally:
        close(handle)
    protected = args.output.parent/"protected_retry_fixture_001.json"
    ORIGINAL(protected, {"value": 0})
    handle, close = hold_read(protected)
    try:
        count = len(operator.trace["events"])
        try:
            operator(protected, {"value": 1})
            checks["outside_owned_tree_original_error_without_retry"] = False
        except PermissionError as error:
            checks["outside_owned_tree_original_error_without_retry"] = error.winerror == 5 and len(operator.trace["events"]) == count and json.loads(protected.read_text()) == {"value": 0}
    finally:
        close(handle)
    try:
        operator(args.output/"invalid.json", {"value": float("nan")})
        checks["original_serialization_error_not_retried"] = False
    except ValueError:
        checks["original_serialization_error_not_retried"] = len(operator.trace["events"]) == 2
    with owned_atomic_json(args.output) as trace:
        checks["only_runtime_alias_bound_original_definition_intact"] = runtime.atomic_json is not ORIGINAL
        runtime.atomic_json(args.output/"context.json", {"value": 3})
    checks["all_module_aliases_restored"] = trace["binding_restored"] and runtime.atomic_json is ORIGINAL
    checks["all_sources_unchanged"] = identities == [data_identity(p) for p in sources]
    ORIGINAL(args.output/"RESULT.json", {"status": "passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": identities, "retry_events": operator.trace["events"], "trace": trace,
        "whole_seconds": time.perf_counter()-started, "counted": False,
        "historical_actual_reader_identity": "unconfirmed"})
    print(json.dumps({"passed": all(checks.values()), "checks": len(checks), "counted": False}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
