"""Apply the declared renderer-interface recovery only after a strict side failure drains."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from atomic_json_windows import atomic_json, read_json


def exact_process_alive(pid, birth):
    try:
        return psutil.Process(pid).create_time() == birth
    except psutil.NoSuchProcess:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    root = code.parents[1]
    args.campaign.resolve().relative_to((root / "04_outputs/gaussian_full_raw_ab_20261008").resolve())
    args.output.resolve().relative_to(args.campaign.resolve())
    cases = read_json(args.campaign / "CASES.json")["cases"]
    if args.case not in {row["case"] for row in cases}:
        raise ValueError("Unknown frozen raw recording")
    args.output.mkdir(parents=True, exist_ok=False)
    side = args.campaign / args.case / args.mode
    state = {"status": "waiting_for_existing_side_outcome", "pid": os.getpid(),
        "process_birth": psutil.Process().create_time(), "case": args.case, "mode": args.mode,
        "recovery_policy": "original_native_Camera_near_plane_with_unchanged_complete_oracles",
        "original_metric_and_track_failures_remain_failed": True,
        "maximum_wait_seconds": 21600, "thresholds_relaxed": False}
    started = time.monotonic()
    try:
        while True:
            previous = read_json(side / "PIPELINE_STATE.json")
            state.update(wait_seconds=time.monotonic() - started, observed_side_stage=previous.get("current_stage"),
                observed_side_status=previous["status"])
            atomic_json(args.output / "STATE.json", state)
            if previous["status"] == "running":
                if not exact_process_alive(previous["pid"], previous["process_birth"]):
                    raise RuntimeError("A stale running side needs a separate controller diagnosis")
            else:
                if exact_process_alive(previous["pid"], previous["process_birth"]):
                    time.sleep(.25)
                    continue
                if previous["status"] != "failed_pipeline_stage" or previous.get("current_stage") != "metric_sim3_track_recovery":
                    state.update(status="side_resolved_without_applicable_projection_recovery", side_outcome=previous)
                    break
                original = read_json(side / "metric/RESULT.json")
                strict = read_json(side / "metric_sim3_track_recovery_process.json")
                oracle = read_json(side / "metric_sim3_track_recovery/ORACLE.json")
                if (original["status"] != "failed_projection_validation" or strict["status"] != "failed_child"
                        or oracle["maximum_projection_delta_px"] <= 1e-6
                        or oracle["maximum_metric_center_delta_m"] > 1e-8):
                    raise ValueError("The observed failure is outside the declared similarity numerical recovery")
                if (side / "metric_sim3_renderer_recovery").exists():
                    raise FileExistsError("A previous renderer-interface recovery must not be duplicated")
                for row in strict["owned_process_identities"]:
                    if exact_process_alive(row["pid"], row["creation_time"]):
                        raise RuntimeError("A failed strict stage still has an exact owned live process")
                state.update(status="running_declared_renderer_interface_recovery", original_oracle=oracle,
                    original_side_failure=previous, all_exact_strict_stage_processes_drained=True)
                atomic_json(args.output / "STATE.json", state)
                command = [sys.executable, str(code / "run_side.py"), "--case", args.case,
                    "--mode", args.mode, "--campaign", str(args.campaign), "--render-near-recovery"]
                with (args.output / "continuation.stdout.log").open("x") as stdout, (args.output / "continuation.stderr.log").open("x") as stderr:
                    child = subprocess.run(command, stdout=stdout, stderr=stderr,
                        env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
                state.update(status="declared_recovery_outcome_recorded", returncode=child.returncode,
                    final_side_outcome=read_json(side / "PIPELINE_STATE.json"))
                break
            if time.monotonic() - started > state["maximum_wait_seconds"]:
                raise TimeoutError("The observation budget expired; the healthy side was not interrupted")
            time.sleep(.25)
    except BaseException as error:
        state.update(status="failed_projection_recovery_observer_or_execution", error=repr(error))
        atomic_json(args.output / "STATE.json", state)
        raise
    state["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.output / "STATE.json", state)


if __name__ == "__main__":
    main()
