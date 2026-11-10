"""Exercise real Windows delete-sharing failures and atomic JSON visibility."""

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import threading
import time

from atomic_json_windows import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json as original_atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    path = args.output / "fixture.json"
    atomic_json(path, {"iteration": 0, "payload": [0] * 512})

    def hold():
        handle = create(str(path), 0x80000000, 3, None, 3, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    handle = hold()
    original_failure = None
    try:
        original_atomic_json(path, {"iteration": 1})
    except OSError as error:
        original_failure = {"type": type(error).__name__, "winerror": error.winerror, "errno": error.errno}
    if not original_failure or original_failure["winerror"] != 5:
        close(handle)
        raise AssertionError("The native Windows access-denied replacement was not reproduced")
    releaser = threading.Thread(target=lambda: (time.sleep(0.25), close(handle)))
    releaser.start()
    recovered = atomic_json(path, {"iteration": 2, "payload": [2] * 512})
    releaser.join()
    assert recovered["retries"] > 0 and json.loads(path.read_text())["iteration"] == 2
    handle = hold()
    persistent_failure = None
    try:
        atomic_json(path, {"iteration": 3}, maximum_retry_seconds=0.08)
    except OSError as error:
        persistent_failure = {"type": type(error).__name__, "winerror": error.winerror}
    finally:
        close(handle)
    assert persistent_failure and json.loads(path.read_text())["iteration"] == 2
    errors, reads = [], [0]
    finished = threading.Event()

    def reader():
        while not finished.is_set():
            try:
                value = read_json(path)
                assert value["payload"] == [value["iteration"]] * 512
                reads[0] += 1
                time.sleep(0.001)
            except BaseException as error:
                errors.append({"error": repr(error), "errno": getattr(error, "errno", None),
                               "winerror": getattr(error, "winerror", None)})
                break

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for item in readers:
        item.start()
    try:
        for iteration in range(4, 204):
            atomic_json(path, {"iteration": iteration, "payload": [iteration] * 512})
    finally:
        finished.set()
        for item in readers:
            item.join()
    if not reads[0] or errors:
        atomic_json(args.output / "FAILURE.json", {"status": "failed_concurrent_visibility_probe",
            "valid_reads": reads[0], "reader_errors": errors, "original_failure": original_failure,
            "transient_failure_recovered": recovered, "persistent_failure_not_swallowed": persistent_failure})
        print(json.dumps(json.loads((args.output / "FAILURE.json").read_text())))
        raise AssertionError("Concurrent visibility probe failed")
    leftovers = [item.name for item in args.output.glob("fixture.json.*.tmp")]
    assert not leftovers
    atomic_json(args.output / "RESULT.json", {"status": "completed", "original_failure": original_failure,
        "transient_failure_recovered": recovered, "persistent_failure_not_swallowed": persistent_failure,
        "concurrent_reader_count": 4, "complete_publications": 200, "valid_reads": reads[0],
        "minimum_reader_cycle_delay_seconds": 0.001,
        "reader_errors": errors, "temporary_files_remaining": leftovers,
        "original_production_failure_trigger": "unknown reader; replacement sharing violation reproduced"})
    print(json.dumps(json.loads((args.output / "RESULT.json").read_text())))


if __name__ == "__main__":
    main()
