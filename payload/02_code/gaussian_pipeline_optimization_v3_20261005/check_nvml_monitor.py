"""Compare original CLI fields with NVML during actual bounded CUDA allocations."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
import psutil
from runtime import atomic_json, data_identity
from integer_sift_matching import DistanceWorker
from nvml_memory_monitor_002 import NativeMemory, LIBRARY, cli_memory
from fused_match_node_002 import linux_identity_check


class MemoryWorker(DistanceWorker):
    """Reuse the original bounded stdio framing and stop/wait semantics."""

    def __init__(self, log, resource):
        self.command = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python",
            "-B", "/mnt/c/"+str(CODE/"nvml_load_worker.py").replace("\\", "/")[3:], "--resource",
            "/mnt/c/"+str(resource.resolve()).replace("\\", "/")[3:]]
        self.log = Path(log).open("xb")
        self.stopped = False
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, bufsize=0)
        self.receipt = {"pid": self.process.pid, "creation_time": psutil.Process(self.process.pid).create_time(), "calls": []}
        self.receipt["info"] = self.rpc({"command": "info"})
        self.receipt["info"].pop("payload")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), CODE/"nvml_memory_monitor_002.py", CODE/"nvml_load_worker.py", LIBRARY]
    sources = [data_identity(p) for p in paths]
    used, free = cli_memory()
    if used > 12*1024 or free < 2*1024:
        raise MemoryError("Original device headroom is unavailable")
    native, worker = NativeMemory(), None
    rows = []
    try:
        worker = MemoryWorker(args.output/"worker.log", args.output/"WSL_RESOURCE.json")
        for size in (0, 32*1024**2, 256*1024**2, 512*1024**2):
            worker.rpc({"command": "allocate", "bytes": size})
            before, cli, after = native.mib("v2"), cli_memory(), native.mib("v2")
            rows.append({"held_bytes": size, "before": before, "CLI": cli, "after": after,
                "all_exact_stable_bracket": before == cli == after,
                "same_used_free_budget_decision": (before[0] <= 12*1024 and before[1] >= 2*1024)
                    == (cli[0] <= 12*1024 and cli[1] >= 2*1024)})
            worker.rpc({"command": "release"})
    finally:
        if worker is not None:
            worker.close()
        native.close()
    import json
    resource = json.loads((args.output/"WSL_RESOURCE.json").read_text(encoding="utf-8"))
    linux = linux_identity_check(resource)
    checks = {"all4_actual_CUDA_hold_states_fields_and_budgets_exact": len(rows) == 4
            and all(r["all_exact_stable_bracket"] and r["same_used_free_budget_decision"] for r in rows),
        "old_library_closed_and_double_close_idempotent": native.closed,
        "original_worker_stop_wait_closed": worker.stopped and worker.receipt["returncode"] == 0,
        "native_buffer_CUDA2GiB_and_Linux32GiB": resource["peak_CUDA_reserved_bytes"] <= 2*1024**3 and resource["peak_RSS_bytes"] <= 32*1024**3,
        "actual_owned_Linux_instance_gone": not linux["alive"],
        "complete_sources_unchanged": sources == [data_identity(p) for p in paths]}
    native.close()
    try:
        native.mib("v2")
    except RuntimeError:
        checks["closed_native_session_cannot_be_reused"] = True
    else:
        checks["closed_native_session_cannot_be_reused"] = False
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "rows": rows, "sources": sources, "metadata": native.metadata, "worker": worker.receipt,
        "resource": resource, "Linux_after_wait": linux, "whole_seconds": time.perf_counter()-STARTED, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks", "rows", "whole_seconds")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("A preserved memory-monitor interface failed")


if __name__ == "__main__":
    main()
