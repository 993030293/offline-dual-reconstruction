"""Reproduce a sharing rejection and verify immutable resource journals."""

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import run_bounded_journal_task as journal


def hold_read(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path.resolve()), 0x80000000, 3, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    close = kernel.CloseHandle
    close.argtypes, close.restype = [wintypes.HANDLE], wintypes.BOOL
    return handle, close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = [Path(__file__), Path(journal.__file__)]
    identities = [data_identity(p) for p in sources]
    checks, runs = {}, []
    previous = args.output/"previous.json"
    atomic_json(previous, {"value": 0})
    handle, close = hold_read(previous)
    try:
        try:
            atomic_json(previous, {"value": 1})
            checks["same_original_WinError5_reproduced_with_known_reader"] = False
        except PermissionError as error:
            checks["same_original_WinError5_reproduced_with_known_reader"] = error.winerror == 5
    finally:
        close(handle)
    target = args.output/"new.json"
    journal.atomic_json(target, {"value": 0})
    sample = target.parent/"new_history/sample_000000.json"
    handle, close = hold_read(sample)
    try:
        journal.atomic_json(target, {"value": 1})
        journal.finalize_journals()
        checks["held_historical_snapshot_does_not_block_new_snapshot_and_final"] = json.loads(target.read_text())["value"] == 1
        checks["previous_held_snapshot_immutable"] = json.loads(sample.read_text())["value"] == 0
    finally:
        close(handle)
    journal._JOURNALS.clear()
    for name, maximum, code, expected_status in (("success", 10, "print(2+3)", "completed"),
            ("failure", 10, "raise SystemExit(7)", "failed_child"),
            ("timeout", 1, "import time; time.sleep(10)", "failed_controller")):
        folder = args.output/name
        folder.mkdir()
        command = [sys.executable, "-B", str(Path(journal.__file__)), "--receipt", str(folder/"guard.json"),
            "--log", str(folder/"child.log"), "--maximum-seconds", str(maximum), "--maximum-rss-gib", "8",
            "--", sys.executable, "-B", "-c", code]
        with (folder/"controller.log").open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=30)
        receipt = json.loads((folder/"guard.json").read_text())
        alive = []
        for identity in receipt.get("owned_process_identities", []):
            try:
                process = psutil.Process(identity["pid"])
                if process.create_time() == identity["creation_time"]:
                    alive.append(identity)
            except psutil.NoSuchProcess:
                pass
        samples = sorted((folder/"guard_history").glob("sample_*.json"))
        checks[name+"_original_status_and_complete_drain"] = receipt["status"] == expected_status and not alive
        checks[name+"_final_equals_last_immutable_sample"] = json.loads(samples[-1].read_text()) == receipt
        checks[name+"_original_budget_and_sources_preserved"] = receipt["maximum_seconds"] == maximum and receipt["maximum_RSS_bytes"] == 8*1024**3 and receipt["source"] == data_identity(Path(journal.__file__))
        runs.append({"name": name, "returncode": completed.returncode, "receipt": receipt,
            "snapshot_count": len(samples), "current_alive": alive})
    checks["actual_exit_codes_preserved"] = [r["returncode"] for r in runs][:2] == [0, 7] and runs[2]["returncode"] != 0
    checks["all_sources_unchanged"] = identities == [data_identity(p) for p in sources]
    atomic_json(args.output/"RESULT.json", {"status": "passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": identities, "runs": runs, "whole_seconds": time.perf_counter()-started,
        "counted": False, "cause_boundary": "known-reader mechanism reproduced; actual historical reader identity remains unconfirmed"})
    print(json.dumps({"passed": all(checks.values()), "checks": len(checks), "counted": False}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
