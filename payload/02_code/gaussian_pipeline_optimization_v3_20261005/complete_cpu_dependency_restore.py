"""Check owned worker interfaces, real failures, and a fresh complete recovery."""

import time

STARTED = time.perf_counter()

from pathlib import Path
import sys
import subprocess
import threading
import json

import psutil

from native_cpu_dependency_bootstrap import native_cpu_dependency_bootstrap
from run_cpu_dependency_restore import compare, linux, read, CODE, ROOT, atomic_json, data_identity

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nrc001/r0"


def generic():
    calls = []

    def importer():
        calls.append(threading.get_ident())
        return {"thread_ident": threading.get_ident(), "fixture_only_not_native_Torch": True}

    with native_cpu_dependency_bootstrap(False, importer) as disabled:
        pass
    with native_cpu_dependency_bootstrap(True, importer) as enabled:
        pass
    checks = {
        "disabled_does_not_launch_or_call": not disabled["native_import_calls"] and disabled["owned_worker_joined"],
        "single_owned_worker_differs_from_parent": len(calls) == 1 and calls[0] != threading.get_ident()
            and enabled["owned_worker_joined"] and enabled["native_import_calls"] == 1,
        "returned_worker_result_is_preserved": enabled["actual_import"]["thread_ident"] == calls[0],
    }
    try:
        with native_cpu_dependency_bootstrap(1, importer):
            pass
    except TypeError:
        checks["non_boolean_rejected"] = True
    else:
        checks["non_boolean_rejected"] = False

    def broken():
        raise ValueError("owned_importer_failure")

    try:
        with native_cpu_dependency_bootstrap(True, broken) as failed:
            pass
    except ValueError as error:
        checks["worker_exception_propagated_and_joined"] = str(error) == "owned_importer_failure" and failed["owned_worker_joined"]
    else:
        checks["worker_exception_propagated_and_joined"] = False
    try:
        with native_cpu_dependency_bootstrap(True, importer) as body:
            raise RuntimeError("owned_body_failure")
    except RuntimeError as error:
        checks["body_exception_propagated_and_joined"] = str(error) == "owned_body_failure" and body["owned_worker_joined"]
    else:
        checks["body_exception_propagated_and_joined"] = False
    checks["no_owned_import_workers_remain"] = not any(thread.name.startswith("owned-CPU-dependency") for thread in threading.enumerate())
    return {"checks": checks, "fixtures_not_native_Torch_or_external_object_evidence": True}


def main():
    OWN.mkdir(parents=True, exist_ok=False)
    atomic_json(OWN / "PROCESS_IDENTITY.json", {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()})
    source_names = ("complete_cpu_dependency_restore.py", "fault_cpu_dependency_restore.py",
                    "profile_cpu_dependency_restore.py", "profile_native_restore_cost.py", "native_cpu_dependency_bootstrap.py")
    identities = [data_identity(CODE / name) for name in source_names]
    copies = OWN / "source_copies"
    copies.mkdir()
    import shutil
    for name in source_names:
        shutil.copyfile(CODE / name, copies / name)
    atomic_json(OWN / "CONTRACT.json", {"sources": identities, "V6_net_speed_floor": .02,
        "unchanged_quality_error_recovery_and_original_inputs": True, "new_optimization_count": 0})
    before = time.perf_counter()
    component = generic()
    component["actual_generic_seconds"] = time.perf_counter() - before
    atomic_json(OWN / "GENERIC_RESULT.json", component)
    if not all(component["checks"].values()):
        raise AssertionError("The owned worker interface failed")
    rows = []
    try:
        for label in ("checksum_error", "actual_cancel", "fresh_complete_recovery"):
            folder = OWN / label
            entry = "profile_cpu_dependency_restore.py" if label == "fresh_complete_recovery" else "fault_cpu_dependency_restore.py"
            tail = ["--mode", "bootstrap"] if label == "fresh_complete_recovery" else ["--fault", label]
            command = [sys.executable, "-B", str(CODE / "run_bounded_journal_task.py"),
                "--receipt", str(OWN / (label + "_GUARD.json")), "--log", str(OWN / (label + ".log")),
                "--maximum-seconds", "600", "--maximum-rss-gib", "4", "--", "wsl", "-d",
                "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                linux(CODE / entry), "--output", linux(folder), *tail]
            before = time.perf_counter()
            code = subprocess.run(command).returncode
            seconds = time.perf_counter() - before
            atomic_json(OWN / (label + "_ACTUAL_GUARD_EXIT.json"), {"returncode": code})
            if label == "fresh_complete_recovery":
                checks = compare(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nrc001/c0/t0/incumbent", folder)
                result = read(folder / "RESULT.json")
                checks["actual_fresh_complete_child0"] = code == 0 and all(result["checks"].values())
            else:
                result = read(folder / "FAULT_RESULT.json")
                checks = dict(result["checks"])
                checks["actual_original_failure_not_swallowed"] = code == 1 and result["status"] == "actual_expected_fault_propagated"
            rows.append({"label": label, "checks": checks, "actual_guard_returncode": code, "actual_entry_seconds": seconds})
            atomic_json(OWN / "STATE.json", {"status": "running", "rows": rows, "new_optimization_count": 0})
            print(json.dumps(rows[-1]), flush=True)
            if not all(checks.values()):
                raise AssertionError("An original fault or complete fresh recovery check failed")
        if identities != [data_identity(CODE / name) for name in source_names]:
            raise AssertionError("An actual completion source changed")
        result = {"status": "generic_actual_faults_and_fresh_complete_recovery_passed",
            "checks": {"all_generic_interfaces": all(component["checks"].values()),
                "all_two_original_failures_and_fresh_complete_recovery": len(rows) == 3 and all(all(row["checks"].values()) for row in rows),
                "actual_sources_unchanged": True}, "rows": rows,
            "actual_controller_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0,
            "full_cost_root_review_and_parent_count_pending": True}
        atomic_json(OWN / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": result["checks"]}), flush=True)
    except BaseException as error:
        atomic_json(OWN / "FAILURE.json", {"type": type(error).__name__, "message": str(error), "rows": rows,
            "actual_controller_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
