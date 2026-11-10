"""Inspect cancelled task resources and explicitly recover an abandoned lease."""

import argparse
import datetime
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import uuid

import psutil


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
SHARED = OUT / "coordination_001"
OWNER = SHARED / "LOCK_OWNER.json"
TOKENS = ("parallel_b_007", "parallel_a_004", "gpu_origin_dependency",
          "native_rgb_decode", "inspect_existing_threads.py")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identities(value):
    if isinstance(value, dict):
        if "pid" in value and "creation_time" in value:
            yield int(value["pid"]), float(value["creation_time"])
        for prefix in ("guard", "child"):
            if prefix + "_pid" in value and prefix + "_creation_time" in value:
                yield int(value[prefix + "_pid"]), float(value[prefix + "_creation_time"])
        for nested in value.values():
            yield from identities(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from identities(nested)


def process_state(pid, creation):
    try:
        process = psutil.Process(pid)
        current = process.create_time()
        same = abs(current - creation) < 0.001
        return {"pid": pid, "recorded_creation_time": creation,
                "same_owned_instance_alive": same, "PID_reused": not same}
    except psutil.NoSuchProcess:
        return {"pid": pid, "recorded_creation_time": creation,
                "same_owned_instance_alive": False, "PID_reused": False}


def current_windows_tasks():
    found = []
    for process in psutil.process_iter(["pid", "create_time", "name", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            args = process.info["cmdline"] or []
            if any(any(token in arg for token in TOKENS) for arg in args):
                found.append({"pid": process.pid, "creation_time": process.info["create_time"],
                              "name": process.info["name"], "cmdline": args})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def current_linux_tasks():
    source = """import json,os,psutil
tokens=('parallel_b_007','parallel_a_004','gpu_origin_dependency','native_rgb_decode','inspect_existing_threads.py')
rows=[]
for p in psutil.process_iter(['pid','create_time','name','cmdline']):
 if p.pid==os.getpid(): continue
 try:
  args=p.info['cmdline'] or []
  if any(any(t in a for t in tokens) for a in args):
   rows.append({'pid':p.pid,'creation_time':p.info['create_time'],'name':p.info['name'],'cmdline':args})
 except (psutil.NoSuchProcess,psutil.AccessDenied): continue
print(json.dumps(rows))
"""
    command = ["wsl.exe", "-d", "E26-C2-LIV-20260907-134802", "--",
               "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", source]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                            check=True, timeout=30)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-abandoned-owner", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OUT / "single_chat_transition_001")
    if output.exists():
        raise FileExistsError("Keep transition receipts immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = load(OWNER)
    sources = [OWNER, OUT / "parallel_b_007/campaign_002_lease.json",
               DOC / "parallel_001/b_007/STATUS.json",
               OUT / "gpu_origin_dependency_001/LEASE_DIAGNOSTIC_001.json",
               OUT / "gpu_origin_dependency_001/diagnostic_001/RESOURCE_CLOSE.json"]
    sources.extend(sorted((OUT / "parallel_b_007/campaign_002").glob("*PROCESS_IDENTITY.json")))
    recorded = {}
    for path in sources:
        for pid, creation in identities(load(path)):
            recorded[(pid, creation)] = process_state(pid, creation)
    windows, linux = current_windows_tasks(), current_linux_tasks()
    checks = {"all_recorded_owned_Windows_instances_gone": all(
                  not row["same_owned_instance_alive"] for row in recorded.values()),
              "no_matching_current_Windows_task": not windows,
              "no_matching_current_WSL_task": not linux}
    payload = {"status": "resource_inspection_passed" if all(checks.values()) else "resources_still_active",
               "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "user_requested_single_chat_and_subagent_stop": True,
               "prior_owner_preserved": prior, "checks": checks,
               "record_sources": [str(path) for path in sources],
               "recorded_Windows_instances": list(recorded.values()),
               "current_Windows_tasks": windows, "current_WSL_tasks": linux,
               "normal_child_completion_claimed": False, "processes_killed": [],
               "recovered_owner": None, "new_optimization_count": 0,
               "coverage": "saved PID plus creation-time ledger and current matching task commands; no unsampled transient or whole-machine idle claim"}
    if args.recover_abandoned_owner:
        if not all(checks.values()):
            output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            raise RuntimeError("Do not recover a lease while an owned task still runs")
        if prior.get("owner") != "batch_b_007_campaign_002" or prior.get("status") != "running":
            raise ValueError("Recovery is limited to the exact cancelled B007 second campaign")
        with (SHARED / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                if load(OWNER) != prior:
                    raise RuntimeError("Resource owner changed during inspection")
                # This operational release does not turn interrupted scientific work into success.
                recovered = {**prior, "status": "released",
                             "lease_released_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "release_method": "verified_abandoned_owner_recovery_after_user_stop",
                             "original_child_normal_completion": False,
                             "recovery_receipt": str(output),
                             "original_campaign_receipt_preserved": str(sources[1])}
                temporary = OWNER.with_name(OWNER.name + "." + uuid.uuid4().hex + ".tmp")
                temporary.write_text(json.dumps(recovered, indent=2), encoding="utf-8")
                os.replace(temporary, OWNER)
                payload["recovered_owner"] = recovered
                payload["status"] = "abandoned_lease_recovered_with_original_interruption_preserved"
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks,
                      "recorded_Windows_instance_count": len(recorded), "processes_killed": []}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
