"""Run remaining frozen checks only after the exact formal pair has terminated."""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from atomic_json_windows import atomic_json, read_json
from runtime import data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--playwright-package", type=Path, required=True)
    parser.add_argument("--candidate-completion", type=Path)
    args = parser.parse_args()
    args.pair = args.pair.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    pair = read_json(args.pair / "PAIR_STATE.json")
    owner = {"pid": pair["pid"], "birth": pair["birth"]}
    reference, candidate = [args.pair / name for name in ("formal_reference_001", "formal_sparse_001")]
    started = time.monotonic()
    state = {"status": "waiting_for_exact_formal_pair", "pid": os.getpid(),
             "birth": psutil.Process().create_time(), "formal_owner": owner, "stages": [], "servers": []}
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def publish():
        state["elapsed_seconds"] = time.monotonic()-started
        atomic_json(args.output / "STATE.json", state)

    def execute(name, command, budget):
        state.update(status="running_remaining_checks", current_stage=name)
        publish()
        receipt, log = args.output / (name+"_process.json"), args.output / (name+".log")
        guarded = [sys.executable, OLD / "run_owned_task.py", "--receipt", receipt, "--log", log,
                   "--maximum-seconds", str(budget), "--maximum-rss-gib", "32", "--", *command]
        done = subprocess.run(list(map(str, guarded)), env=env)
        recorded = read_json(receipt)
        state["stages"].append({"stage": name, "status": recorded["status"],
                                "seconds": recorded["elapsed_seconds"], "receipt": str(receipt)})
        publish()
        if done.returncode or recorded["status"] != "completed":
            raise RuntimeError("A remaining check failed: "+name)

    atomic_json(args.output / "CONTRACT.json", {"formal_pair": str(args.pair), "owner": owner,
        "source": data_identity(Path(__file__)), "no_heavy_check_during_formal_pair": True,
        "research_repeats_and_other_input_probe_are_separate_from_each_formal_pipeline_clock": True,
        "formal_acceptance_thresholds_unchanged": True})
    publish()
    try:
        while True:
            try:
                process = psutil.Process(owner["pid"])
                alive = process.is_running() and process.create_time() == owner["birth"]
            except psutil.NoSuchProcess:
                alive = False
            latest = read_json(args.pair / "PAIR_STATE.json")
            if not alive:
                recovered = args.candidate_completion is not None and latest["status"] == "failed_pair_campaign" and \
                    read_json(args.candidate_completion).get("recovered_current_campaign_only") and \
                    read_json(args.candidate_completion)["status"] == "complete_raw_model_preview_and_QA_comparison_pending"
                if latest["status"] != "both_full_raw_runs_complete_quality_and_time_acceptance_pending" and not recovered:
                    raise RuntimeError("The exact pair stopped without two complete formal runs")
                state["verified_formal_owner_live"] = False
                state["verified_formal_pair_terminal_status"] = latest["status"]
                break
            state["verified_formal_owner_live"] = True
            state["formal_route"] = latest.get("current_route")
            publish()
            time.sleep(2)
        state["verified_wait_seconds"] = time.monotonic()-started
        acceptance_command = [sys.executable, CODE / "validate_pair.py", "--reference", reference,
            "--candidate", candidate, "--output", args.output / "paired_acceptance"]
        if args.candidate_completion:
            acceptance_command.extend(["--candidate-completion", args.candidate_completion])
        execute("paired_acceptance", acceptance_command, 180)
        accepted = read_json(args.output / "paired_acceptance/RESULT.json")
        if not accepted["all_primary_gates_passed"]:
            raise ValueError("The frozen formal quality or time gates did not pass")
        execute("matching_repeats", [sys.executable, CODE / "audit_incremental_matching.py",
            "--augmentation", candidate / "box1/optimized/pose_gap", "--output", args.output / "matching_repeats",
            "--repeat", "3"], 660)
        old_box2 = ROOT / "04_outputs/gaussian_full_raw_ab_20261008/box2/optimized"
        cases = read_json(ROOT / "04_outputs/gaussian_full_raw_ab_20261008/CASES.json")["cases"]
        box2_raw = next(row["raw"] for row in cases if row["case"] == "box2")
        probe = args.output / "box2_sparse_probe"
        execute("box2_probe", [sys.executable, CODE / "run_native_sparse.py", "--kind", "temporal", "--mode",
            "optimized", "--raw", box2_raw, "--initial", old_box2 / "initial", "--recovery", old_box2 / "recovery",
            "--output", probe, "--purpose", "pose_gap", "--threads", "8"], 1200)
        probe_result = read_json(probe / "RESULT.json")
        if not (probe_result["original_registration_guard_passed"] and probe_result["pose_gap_registration_guard_passed"]):
            raise ValueError("The Box2 local interface and registration checks failed")
        atomic_json(probe / "DIAGNOSTIC_BOUNDARY.json", {"historical_initial_and_recovery_used": True,
            "initial": str(old_box2 / "initial"), "recovery": str(old_box2 / "recovery"),
            "not_another_fresh_raw_complete_reconstruction_or_training": True,
            "does_not_prove_all_object_generalization": True, "formal_Box1_time_not_replaced": True})
        for route, campaign, port in (("reference", reference, 8226), ("candidate", candidate, 8227)):
            with socket.socket() as socket_probe:
                socket_probe.bind(("127.0.0.1", port))
            stream = (args.output / (route+"_preview_server.log")).open("x", encoding="utf-8")
            command = [sys.executable, OLD / "serve_raw_viewer.py", "--root", campaign / "viewer", "--port", str(port)]
            process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                env=env, creationflags=subprocess.CREATE_NO_WINDOW)
            stream.close()
            url = f"http://127.0.0.1:{port}/box1-optimized/cases/packed/index.html"
            identity = {"route": route, "pid": process.pid, "birth": psutil.Process(process.pid).create_time(),
                        "root": str(campaign / "viewer"), "url": url, "retained_for_review": True}
            state["servers"].append(identity)
            publish()
            deadline = time.monotonic()+15
            while True:
                if process.poll() is not None:
                    raise RuntimeError("A new review preview server stopped before readiness")
                try:
                    with urllib.request.urlopen(url, timeout=1) as reply:
                        if reply.status == 200:
                            break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.1)
            execute(route+"_preview", [sys.executable, OLD / "gpu_lease.py", "--receipt",
                args.output / (route+"_preview_GPU_LEASE.json"), "--label", route+"_after_formal_interaction",
                "--", args.node, CODE / "verify_preview.cjs", args.playwright_package, url,
                args.output / (route+"_preview")], 300)
        state["status"] = "remaining_computational_checks_complete_visual_and_text_review_pending"
        publish()
        atomic_json(args.output / "RESULT.json", state)
        print(state["status"], flush=True)
    except BaseException:
        state.update(status="failed_remaining_checks_preserved", traceback=traceback.format_exc())
        publish()
        atomic_json(args.output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
