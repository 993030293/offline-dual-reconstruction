"""Interrupt one owned parallel task and verify cancellation before publication."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    proof = read(args.campaign/"RESULT.json")
    if not all(proof["checks"].values()):
        raise ValueError("The measured complete DAG protection is missing")
    protocol = read(args.campaign/"trial0/parallel/PROTOCOL.json")
    checkpoint = next(r["path"] for r in protocol["input"] if r["path"].endswith(".pth"))
    localization = ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330/05_evaluation/C_augmented_212/H_LOCALIZATION.json"
    command = [sys.executable, str(Path(__file__).with_name("posttrain_dag.py")), "--mode", "parallel",
        "--pair-source", protocol["same_actual_training_recovery_run"], "--checkpoint", checkpoint,
        "--runtime", str(ROOT/"04_outputs/box2_3dgs_display_v1_20260930"),
        "--reference-run", str(ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330"),
        "--localization", str(localization), "--output", str(args.output/"failed_DAG")]
    with (args.output/"controller.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned = []
        try:
            start = time.monotonic()
            while time.monotonic()-start < 20:
                if process.poll() is not None:
                    raise RuntimeError("The owned fault-test DAG completed before an actual injected failure")
                path = args.output/"failed_DAG/STATE.json"
                state = read(path) if path.exists() else {}
                tasks = state.get("tasks", {})
                receipts = [args.output/"failed_DAG"/(task+"_process.json") for task in ("evaluate", "archive")]
                if set(tasks) == {"evaluate", "archive"} and all(p.exists() for p in receipts):
                    records = [read(p) for p in receipts]
                    if all(r["status"] == "running" and r.get("child_pid") for r in records):
                        controllers = [psutil.Process(tasks[t]["controller_pid"]) for t in ("evaluate", "archive")]
                        for controller in controllers:
                            owned += [{"pid": p.pid, "creation_time": p.create_time()} for p in [controller, *controller.children(recursive=True)]]
                        # Signal only the new owned task's verified session group.
                        evaluation = controllers[0]
                        if os.getpgid(evaluation.pid) != evaluation.pid or not controllers[1].is_running():
                            raise ValueError("The owned task process group or live peer identity changed")
                        os.killpg(evaluation.pid, signal.SIGTERM)
                        break
                time.sleep(.02)
            else:
                raise TimeoutError("An actual owned pair of live DAG tasks was not observed")
            code = process.wait(timeout=20)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
    state = read(args.output/"failed_DAG/STATE.json")
    terminated = []
    for row in owned:
        try:
            candidate = psutil.Process(row["pid"])
            absent = candidate.create_time() != row["creation_time"] or not candidate.is_running() or candidate.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            absent = True
        terminated.append(absent)
    checks = {"actual_task_group_failure_propagated": code != 0,
        "no_incomplete_publication": not (args.output/"failed_DAG/PUBLICATION.json").exists()
            and not state["publication_created"] and state["status"] == "failed_no_publication",
        "both_owned_task_groups_no_longer_running": bool(owned) and all(terminated),
        "measured_DAG_sources_unchanged": proof["sources"] == [data_identity(Path(r["path"])) for r in proof["sources"]]}
    result = {"status": "actual_parallel_failure_and_cancellation_guards_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "owned_task_identities": owned, "terminated_or_zombie": terminated,
        "unrelated_processes_or_existing_services_modified": False, "counted": False,
        "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": checks})
    if not all(checks.values()):
        raise ValueError("An actual concurrent DAG cancellation guard failed")


if __name__ == "__main__":
    main()
