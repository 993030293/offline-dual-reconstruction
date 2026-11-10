"""Keep the primary lease until original guards and actual owned workers drain."""

import argparse
import ctypes
from ctypes import wintypes
import datetime
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from fused_match_node_config import CODE, DOC, OUT, V1, read, atomic_json, data_identity


def safe_json(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(descriptor, "r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def remaining(identities):
    answer = []
    for row in identities:
        try:
            process = psutil.Process(row["pid"])
            if process.create_time() == row["creation_time"]:
                answer.append(row)
        except psutil.NoSuchProcess:
            pass
    return answer


def linux_identity_check(identity, drain=False):
    payload = json.dumps({"pid": identity["pid"], "creation_time": identity["creation_time"]})
    code = "import json,psutil; i=json.loads("+repr(payload)+"); result=dict(i,alive=False,cmdline=[])\n"
    code += "try:\n p=psutil.Process(i['pid']); result.update(alive=p.create_time()==i['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE,cmdline=p.cmdline())\nexcept psutil.NoSuchProcess: pass\n"
    if drain:
        entries = ("parallel_a_003/owned_matching_worker.py", "fused_owned_matching_worker.py", "check_fused_integer_reduction.py")
        allowed = tuple("/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/"+name for name in entries)
        code += "if result['alive']:\n allowed="+repr(allowed)+"\n assert any(x in result['cmdline'] for x in allowed), 'An unrelated Linux PID cannot be drained'\n p.terminate()\n try: p.wait(timeout=10)\n except psutil.TimeoutExpired: p.kill(); p.wait(timeout=10)\n result.update(alive=False,forced_owned_drain=True)\n"
    code += "print(json.dumps(result))"
    return json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["localization", "training", "camera_oracle", "quality", "recovery_check"], required=True)
    parser.add_argument("--mode", choices=["incumbent", "fused"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for path in (args.node, args.output):
        path.resolve().relative_to(OUT.resolve())
    args.node.mkdir(parents=True, exist_ok=False)
    if args.output.exists():
        raise FileExistsError(args.output)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A concrete original protected child command is required")
    source = data_identity(Path(__file__))
    shutil.copyfile(Path(__file__), args.node/"OWNED_CONTROLLER_SOURCE.py")
    tracked = {}
    peaks = {"Windows_tree_RSS_bytes": 0, "GPU_total_used_bytes": 0, "owned_output_bytes": 0}
    record = {"status": "starting", "kind": args.kind, "mode": args.mode, "seed": args.seed,
              "source": source, "command": command, "primary_lease_owned_by_outer_launcher": True,
              "Linux_RSS_is_separate_from_Windows": True, "counted": False}
    bounded = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.node/"guard.json"),
        "--log", str(args.node/"child.log"), "--maximum-seconds", "1750", "--maximum-rss-gib", "8", "--", *command]

    def publish(owned):
        atomic_json(args.node/"NODE.json", record)

    with (args.node/"controller.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(bounded, stdout=log, stderr=subprocess.STDOUT)
        owner = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
        record["owner"] = owner
        publish([owner])
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(owner["pid"])
                    if parent.create_time() != owner["creation_time"]:
                        raise RuntimeError("The original guard PID identity changed")
                    processes = [parent, *parent.children(recursive=True)]
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    raise
                for item in processes:
                    try:
                        tracked[item.pid] = item.create_time()
                    except psutil.NoSuchProcess:
                        pass
                owned = remaining([{"pid": pid, "creation_time": born} for pid, born in tracked.items()])
                rss = 0
                for row in owned:
                    try:
                        item = psutil.Process(row["pid"])
                        if item.create_time() == row["creation_time"]:
                            rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
                used, free = map(int, gpu.strip().split(","))
                size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
                peaks = {"Windows_tree_RSS_bytes": max(peaks["Windows_tree_RSS_bytes"], rss),
                         "GPU_total_used_bytes": max(peaks["GPU_total_used_bytes"], used*1024**2),
                         "owned_output_bytes": max(peaks["owned_output_bytes"], size)}
                record.update(status="running", resource_peaks=peaks)
                publish(owned)
                if rss > 8*1024**3 or used > 12*1024 or free < 2*1024 or size > 30*1024**3 or shutil.disk_usage(OUT).free < 20*1024**3 or time.perf_counter()-started > 1770:
                    raise RuntimeError("A frozen node resource bound was exceeded")
                time.sleep(1)
        except BaseException as error:
            record["controller_error"] = repr(error)
            victims = []
            for row in reversed(remaining([{"pid": pid, "creation_time": born} for pid, born in tracked.items()])):
                try:
                    item = psutil.Process(row["pid"])
                    if item.create_time() == row["creation_time"] and row != owner:
                        item.kill()
                        victims.append(item)
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(victims, timeout=10)
            process.wait(timeout=1780)
        record["returncode"] = process.wait()
    guard = read(args.node/"guard.json")
    identities = list(guard.get("owned_process_identities", []))+[owner]+[{"pid": pid, "creation_time": born} for pid, born in tracked.items()]
    linux_receipts, linux_checks = [], []
    for path in [*args.output.rglob("WSL_RESOURCE.json"), *args.output.glob("*PROCESS_IDENTITY.json")]:
        value = read(path)
        if value.get("scope") == "WSL" or path.name == "WSL_RESOURCE.json":
            linux_receipts.append({"path": str(path), **value})
            actual = linux_identity_check(value)
            if actual["alive"]:
                record["controller_error"] = record.get("controller_error") or "An owned Linux process remained after the original guard wait"
                actual = linux_identity_check(value, drain=True)
            linux_checks.append(actual)
    record.update(status="completed" if record["returncode"] == 0 and not record.get("controller_error") else "failed",
                  guard=guard, live_Windows_after_wait=remaining(identities), actual_Linux_after_wait=linux_checks,
                  Linux_resource_receipts=linux_receipts, resource_peaks=peaks,
                  original_guard_whole_seconds=guard["elapsed_seconds"], complete_node_seconds=time.perf_counter()-started,
                  final_source_unchanged=source == data_identity(Path(__file__)))
    publish(record["live_Windows_after_wait"])
    if record["returncode"] or record.get("controller_error") or record["live_Windows_after_wait"] or any(row["alive"] for row in linux_checks) or not record["final_source_unchanged"] or record["complete_node_seconds"] > 1800:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
