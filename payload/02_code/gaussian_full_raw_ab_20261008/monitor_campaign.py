"""Sample existing NVML and host memory while the verified campaign controller lives."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005")]
from runtime import atomic_json, data_identity
from nvml_memory_monitor_002 import NativeMemory


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    controller = read(args.campaign / "CAMPAIGN_STATE.json")
    native = NativeMemory()
    contract = {"pid": os.getpid(), "source": data_identity(Path(__file__)), "NVML": native.metadata,
        "controller_pid": controller["pid"], "controller_birth": controller["process_birth"],
        "cadence_seconds": 2, "started_unix": time.time(), "samples_before_start_not_available": True,
        "GPU_scope": "device-global usage includes Windows desktop and unrelated apps; not per-job allocation",
        "training_Torch_peak_reserved_is_separately_recorded": True}
    atomic_json(args.output / "CONTRACT.json", contract)
    samples = 0
    maximum = {"GPU_global_used_bytes": 0, "host_used_bytes": 0}
    try:
        with (args.output / "samples.jsonl").open("x", encoding="utf-8") as stream:
            while True:
                current = read(args.campaign / "CAMPAIGN_STATE.json")
                if current["pid"] != controller["pid"] or current["process_birth"] != controller["process_birth"]:
                    raise ValueError("Campaign controller ownership changed during sampling")
                try:
                    process = psutil.Process(controller["pid"])
                    alive = process.create_time() == controller["process_birth"]
                except psutil.NoSuchProcess:
                    alive = False
                if current["status"] != "running" or not alive:
                    break
                memory = psutil.virtual_memory()
                gpu = native.read("v2")
                side_path = args.campaign / current["current_case"] / current["current_mode"] / "PIPELINE_STATE.json"
                side = read(side_path) if side_path.exists() else {}
                row = {"unix_seconds": time.time(), "case": current["current_case"], "mode": current["current_mode"],
                    "stage": side.get("current_stage"), "host_used_bytes": memory.total-memory.available,
                    "host_total_bytes": memory.total, "GPU_global_memory": gpu}
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                samples += 1
                maximum["GPU_global_used_bytes"] = max(maximum["GPU_global_used_bytes"], gpu["used"])
                maximum["host_used_bytes"] = max(maximum["host_used_bytes"], row["host_used_bytes"])
                if samples % 10 == 0:
                    atomic_json(args.output / "STATE.json", {"status": "sampling", "samples": samples, "last": row})
                time.sleep(2)
        atomic_json(args.output / "RESULT.json", {"status": "controller_sampling_completed", "samples": samples,
            "maximum_global_samples": maximum, "independent_job_peak_claim": False,
            "sampling_interval_is_not_continuous_peak_measurement": True})
    finally:
        native.close()


if __name__ == "__main__":
    main()
