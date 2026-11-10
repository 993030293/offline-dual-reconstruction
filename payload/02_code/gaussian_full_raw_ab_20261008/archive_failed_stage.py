"""Preserve a verified dead failed stage before an explicit independent retry."""

import argparse
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    side = args.side.resolve()
    side.relative_to((ROOT / "04_outputs/gaussian_full_raw_ab_20261008").resolve())
    if args.stage not in ("pose_gap", "temporal", "metric", "initial", "sfm", "export", "training", "coarse", "recovery"):
        raise ValueError("Unknown owned stage")
    receipt_path = side / (args.stage+"_process.json")
    receipt = read(receipt_path)
    if receipt["status"] not in ("failed_child", "failed_controller", "failed_RSS_guard", "failed_time_guard"):
        raise ValueError("Only an explicitly failed stage may be archived")
    identities = receipt["owned_process_identities"]
    controller_path = side / "PIPELINE_STATE.json"
    if controller_path.exists():
        controller = read(controller_path)
        identities = [*identities, {"pid": controller["pid"], "creation_time": controller["process_birth"]}]
    for row in identities:
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                raise RuntimeError("An exact owned stage or controller process remains alive")
        except psutil.NoSuchProcess:
            pass
    tag = str(time.time_ns())
    archive = side / ("failed_"+args.stage+"_"+tag)
    archive.mkdir(exist_ok=False)
    paths = [side / args.stage, receipt_path, side / (args.stage+".log"), side / (args.stage+"_wrapper.json")]
    if args.stage == "training":
        paths += [side / "training_execution_sources", side / "training_execution_contract.json"]
    paths += sorted(side.glob(args.stage+"_GPU_LEASE_*.json"))
    checked = []
    for path in paths:
        if path.exists():
            path.resolve().relative_to(side)
            if path.is_symlink():
                raise ValueError("An owned stage archive may not move a symbolic link")
            checked.append(path)
    record = {"stage": args.stage, "reason": args.reason, "receipt": receipt,
              "archive": str(archive), "all_exact_owned_processes_dead": True,
              "elapsed_seconds": receipt["elapsed_seconds"], "moved_paths": [str(path) for path in checked],
              "controller_state_present": controller_path.exists()}
    atomic_json(archive / "ARCHIVE.json", record)
    for path in checked:
        path.rename(archive / path.name)
    atomic_json(side / ("FAILED_STAGE_ATTEMPT_"+tag+".json"), record)
    print(json.dumps({"archive": str(archive), "failed_seconds_preserved": receipt["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
