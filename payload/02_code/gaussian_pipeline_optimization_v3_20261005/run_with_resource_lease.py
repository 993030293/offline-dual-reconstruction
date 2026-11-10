"""Serialize complete measured campaigns across the authorized local chats."""

import argparse
import datetime
import json
import msvcrt
import os
import subprocess
import time
import uuid
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"


def save(path, payload):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def validate_prior_owner(path):
    if not path.exists():
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    if prior.get("status") != "released" or not prior.get("lease_released_utc"):
        raise RuntimeError(
            "Prior resource owner did not record a complete drain/release; "
            "inspect its Windows and WSL descendants before another campaign"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    receipt.relative_to(ROOT / "04_outputs")
    if receipt.exists():
        raise FileExistsError("A lease receipt must have a fresh path")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or not 0 < args.wait_seconds <= 3600:
        raise ValueError("A bounded wait and a concrete child command are required")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(SHARED / "exclusive_benchmark.lock", os.O_CREAT | os.O_RDWR)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b"\x00")
    started = time.perf_counter()
    acquired = False
    payload = {
        "owner": args.owner, "guard_pid": os.getpid(),
        "guard_creation_time": psutil.Process().create_time(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "all_measured_CPU_RAW_GPU_campaigns",
        "lock": str(SHARED / "exclusive_benchmark.lock"),
        "lock_wait_in_algorithm_cost": False,
    }
    process = None
    try:
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
                break
            except OSError as error:
                if error.errno not in (13, 36):
                    raise
                if time.perf_counter() - started >= args.wait_seconds:
                    raise TimeoutError("Another measured campaign retains its resource lease")
                time.sleep(0.25)
        validate_prior_owner(SHARED / "LOCK_OWNER.json")
        payload["wait_seconds"] = time.perf_counter() - started
        payload["status"] = "running"
        save(SHARED / "LOCK_OWNER.json", payload)
        save(receipt, payload)
        child_started = time.perf_counter()
        process = subprocess.Popen(command, cwd=ROOT)
        payload["child_pid"] = process.pid
        payload["child_creation_time"] = psutil.Process(process.pid).create_time()
        save(SHARED / "LOCK_OWNER.json", payload)
        save(receipt, payload)
        return_code = process.wait()
        payload["child_elapsed_seconds"] = time.perf_counter() - child_started
        payload["returncode"] = return_code
        payload["status"] = "child_exited"
        save(receipt, payload)
    except BaseException as error:
        payload["status"] = "lease_or_child_failed"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        if process is not None and process.poll() is None:
            # The child must finish before its resources can be released.
            payload["child_pid"] = process.pid
            payload["child_returncode_after_drain"] = process.wait()
        save(receipt, payload)
        raise
    finally:
        if acquired:
            payload["lease_released_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save(receipt, payload)
            if payload.get("wait_seconds") is not None:
                save(SHARED / "LOCK_OWNER.json", {**payload, "status": "released"})
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
