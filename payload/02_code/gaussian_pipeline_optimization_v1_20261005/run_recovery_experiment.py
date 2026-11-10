"""Exercise a real process interruption and complete-state continuation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsplat-build", type=Path)
    parser.add_argument("--iterations", type=int, choices=[1600, 12000], default=1600)
    parser.add_argument("--vectorized-ply", action="store_true")
    parser.add_argument("--gpu-resident-inputs", action="store_true")
    parser.add_argument("--residency-acceptance", type=Path)
    args = parser.parse_args()
    if args.gpu_resident_inputs:
        if not args.gsplat_build or not args.residency_acceptance:
            raise ValueError("Combined native GPU inputs require accepted residency and the verified backend")
        acceptance = json.loads(args.residency_acceptance.read_text())
        if (acceptance["status"] != "accepted_stage_optimization"
                or acceptance["mechanism"] != "P26_INPUT_RESIDENCY"
                or acceptance["fixed_seeds"] != [0, 1, 2]
                or not all(acceptance["checks"].values())):
            raise ValueError("P26 did not pass its complete independent stage protocol")
    if args.residency_acceptance and not args.gpu_resident_inputs:
        raise ValueError("A residency acceptance cannot silently select CPU inputs")
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise MemoryError("Insufficient free disk for the isolated recovery group")
    interval, crash_at = (900, 935) if args.iterations == 1600 else (4000, 8035)
    resume_at = (crash_at // interval) * interval
    maximum_seconds = 180 if args.iterations == 1600 else 1800
    args.output.mkdir(parents=True, exist_ok=False)
    source_root = args.output/"frozen_scripts"
    source_root.mkdir()
    names = ["train_case.py", "runtime.py", "full_intrinsics.py", "checkpoint_adapter.py"]
    if args.gsplat_build:
        names.append("gsplat_adapter.py")
    if args.vectorized_ply:
        names.append("ply_serialization.py")
    if args.gpu_resident_inputs:
        names.append("host_memory.py")
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), source_root/name)
    state = {"status": "running", "pid": os.getpid(), "runs": [], "source": data_identity(Path(__file__)),
        "iterations": args.iterations, "checkpoint_interval": interval,
        "checkpoint_at": resume_at, "crash_at": crash_at, "expected_lost_steps": crash_at-resume_at,
        "acceptance": "exact restored state and sampling; real exit86 and continuation; development quality guard and visual check"}
    state.update(backend="gsplat" if args.gsplat_build else "reference",
        gsplat_build=data_identity(args.gsplat_build/"RESULT.json") if args.gsplat_build else None,
        sources=[data_identity(source_root/name) for name in names], new_optimization_count=0,
        combination_scope="P0_full_K_and_P3_complete_recovery_and_P6_gsplat" if args.gsplat_build else "original_P3")
    state.update(vectorized_ply=args.vectorized_ply, native_byte_oracle=args.vectorized_ply,
        gpu_resident_inputs=args.gpu_resident_inputs, maximum_seconds_per_process=maximum_seconds,
        residency_acceptance=data_identity(args.residency_acceptance) if args.residency_acceptance else None)
    state["combination_mechanisms"] = ["P0_FULL_K", "P3_COMPLETE_TRAINING_RECOVERY"]
    if args.gsplat_build:
        state["combination_mechanisms"].append("P6_GSPLAT_BACKEND")
    if args.vectorized_ply:
        state["combination_mechanisms"].append("P22_VECTORIZED_PLY")
    if args.gpu_resident_inputs:
        state["combination_mechanisms"].append("P26_INPUT_RESIDENCY")
    started = time.monotonic()
    atomic_json(args.output/"STATE.json", state)
    for label in ["continuous", "interrupted", "resumed_interrupted", "resumed"]:
        command = [sys.executable, str(source_root/"train_case.py"),
            "--runtime", str(args.runtime), "--dataset", str(args.dataset), "--output", str(args.output/label),
            "--projection", "full", "--seed", "0", "--iterations", str(args.iterations),
            "--checkpoint-every", str(interval), "--maximum-seconds", str(maximum_seconds)]
        if args.gsplat_build:
            command += ["--gsplat-build", str(args.gsplat_build)]
        if args.vectorized_ply:
            command += ["--vectorized-ply", "--verify-native-ply"]
        if args.gpu_resident_inputs:
            command += ["--gpu-resident-inputs", "--trim-host-after-load"]
        if label == "interrupted":
            command += ["--simulate-crash-after", str(crash_at)]
        if label.startswith("resumed"):
            source = "interrupted" if label == "resumed_interrupted" else "continuous"
            command += ["--resume-file", str(args.output/source/f"checkpoint_{resume_at:06d}.pth")]
        row = {"variant": label, "command": command, "status": "starting"}
        state["runs"].append(row)
        with (args.output/(label+".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row.update(status="running", pid=process.pid)
            atomic_json(args.output/"STATE.json", state)
            try:
                code = process.wait(timeout=maximum_seconds+100)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                row["status"] = "watchdog_timeout"
                state["status"] = "failed"
                atomic_json(args.output/"STATE.json", state)
                raise
        expected_code = 86 if label == "interrupted" else 0
        row.update(returncode=code, status="expected_crash" if code == 86 else "completed" if code == 0 else "failed")
        if label == "interrupted" and code == 86:
            child_state_path = args.output/label/"STATE.json"
            child_state = json.loads(child_state_path.read_text())
            child_state.update(status="crashed_as_injected", controller_observed_exitcode=86,
                               process_terminated=True, iteration_at_exit=crash_at)
            atomic_json(child_state_path, child_state)
        atomic_json(args.output/"STATE.json", state)
        if code != expected_code:
            state["status"] = "failed"
            atomic_json(args.output/"STATE.json", state)
            raise RuntimeError("Recovery experiment exited differently from its registered contract")
        if sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) > 30 * 1024**3:
            state.update(status="failed", error="Registered group disk budget exceeded")
            atomic_json(args.output/"STATE.json", state)
            raise MemoryError(state["error"])
    trace = lambda label: json.loads((args.output/label/"SAMPLING_TRACE.json").read_text())["names"]
    checks = {"continuous_sample_count_matches_registered_budget": len(trace("continuous")) == args.iterations,
        "same_checkpoint_continuation_sampling_exact": trace("continuous") == trace("resumed"),
        "crash_continuation_sampling_exact": trace("continuous") == trace("resumed_interrupted"),
        "crashed_process_did_not_publish_final_model": not (args.output/f"interrupted/model/point_cloud/iteration_{args.iterations}/point_cloud.ply").exists()}
    for label in ["resumed", "resumed_interrupted"]:
        restored = json.loads((args.output/label/"RESTORE_CHECK.json").read_text())
        checks[label+"_state_exact"] = restored["status"] == "all_saved_state_exact_before_next_training_iteration" and all(restored["checks"].values())
        legacy = json.loads((args.output/label/"LEGACY_SAMPLER_ORACLE.json").read_text())
        checks[label+"_legacy_sampler_fails_oracle"] = not legacy["native_preserves_sampling"]
    for label in ["continuous", "resumed", "resumed_interrupted"]:
        child = json.loads((args.output/label/"STATE.json").read_text())
        checks[label+"_completed_budget"] = child["status"] == "completed" and child["iteration"] == args.iterations
        if args.vectorized_ply:
            saved = json.loads((args.output/label/"PLY_SERIALIZATION.json").read_text())
            checks[label+"_all_native_ply_bytes_exact"] = bool(saved["events"]) and all(
                event["native_byte_oracle"]["identical"] for event in saved["events"])
            checks[label+"_correct_process_save_schedule"] = (saved["schedule_validation"]["expected_process_iterations"]
                == saved["schedule_validation"]["observed_process_iterations"])
        if args.gpu_resident_inputs:
            residency = child["scene_timing"]["input_residency"]
            checks[label+"_212_GPU_inputs_without_CPU_image_copy"] = (residency["camera_count"] == 212
                and residency["payload_bytes_by_device"] == {"cpu": 0, "cuda": 1707668480})
    state.update(status="state_checks_passed_quality_pending" if all(checks.values()) else "failed_state_checks",
                 checks=checks, elapsed_seconds=time.monotonic()-started, counted=False)
    atomic_json(args.output/"STATE.json", state)
    print(json.dumps({"status": state["status"], "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
