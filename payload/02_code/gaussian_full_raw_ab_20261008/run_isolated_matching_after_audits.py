"""Repeat complete matching after owned reconstruction and localization work drains."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from atomic_json_windows import atomic_json, read_json


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import data_identity


def blockers(campaign):
    paths = [campaign / "CAMPAIGN_STATE.json", campaign / "EVALUATION_CAMPAIGN_STATE.json"]
    paths.extend(campaign.glob("*/*/PIPELINE_STATE.json"))
    paths.extend(campaign.glob("*/localization_modules/STATE.json"))
    pending = []
    for path in paths:
        if not path.is_file():
            pending.append({"path": str(path), "status": "missing_required_state"})
            continue
        state = read_json(path)
        if state.get("status") in ("running", "waiting_for_primary_if_requested"):
            pending.append({"path": str(path), "status": state["status"]})
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    campaign.relative_to((ROOT / "04_outputs/gaussian_full_raw_ab_20261008").resolve())
    args.output.resolve().relative_to(campaign)
    if args.check_only:
        import json
        print(json.dumps({"blockers": blockers(campaign), "no_compute_launched": True}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = campaign / "box2/matching_modules_isolated"
    if result_path.exists():
        raise FileExistsError("An isolated native repetition must preserve all previous outputs")
    audit = CODE / "audit_sfm_matching.py"
    frozen_source = data_identity(audit)
    state = {"status": "waiting_for_owned_primary_and_localization",
        "pid": os.getpid(), "process_birth": psutil.Process().create_time(),
        "source": data_identity(Path(__file__)), "audit_source": frozen_source,
        "maximum_wait_seconds": 21600, "matching_threads": [8, 8],
        "no_primary_source_or_quality_policy_changes": True}
    started = time.monotonic()
    try:
        while True:
            waiting = blockers(campaign)
            state.update(blockers=waiting, wait_seconds=time.monotonic() - started)
            atomic_json(args.output / "STATE.json", state)
            if not waiting:
                break
            if time.monotonic() - started > state["maximum_wait_seconds"]:
                raise TimeoutError("Owned computations have not drained; matching was not started")
            time.sleep(5)
        if data_identity(audit) != frozen_source:
            raise ValueError("The queued audit source changed before execution")
        command = [sys.executable, str(CODE / "run_owned_task.py"),
            "--receipt", str(args.output / "matching_process.json"),
            "--log", str(args.output / "matching.log"),
            "--maximum-seconds", "2400", "--maximum-rss-gib", "12", "--",
            sys.executable, str(audit), "--features",
            str(campaign / "box2/optimized/sfm/features.db"), "--pairs",
            str(campaign / "box2/optimized/sfm/guided_pairs.txt"),
            "--output", str(result_path), "--require-isolated"]
        state.update(status="running_isolated_complete_matching", command=command,
            owned_primary_and_localization_receipts_terminal=True)
        atomic_json(args.output / "STATE.json", state)
        child = subprocess.run(command, env={**os.environ, "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1"})
        if child.returncode or not (result_path / "RESULT.json").is_file():
            raise RuntimeError("The isolated complete matching audit failed; outputs retained")
        state.update(status="completed_isolated_matching_outcomes", result=read_json(result_path / "RESULT.json"))
    except BaseException as error:
        state.update(status="failed_isolated_matching_schedule_or_execution", error=repr(error))
        atomic_json(args.output / "STATE.json", state)
        raise
    state["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.output / "STATE.json", state)


if __name__ == "__main__":
    main()
