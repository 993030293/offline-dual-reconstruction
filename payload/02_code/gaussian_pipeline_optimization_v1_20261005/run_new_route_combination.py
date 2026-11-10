"""Run the registered new-map control and actual crash/resume combination."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--recovery-acceptance", type=Path, required=True)
    parser.add_argument("--gsplat-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    smoke = read(args.smoke / "STATE.json")
    smoke_config = read(args.smoke / "CONFIG.json")
    smoke_inputs = read(args.smoke / "INPUT_MANIFEST.json")
    recovery = read(args.recovery_acceptance)
    admission = read(args.admission)
    if not (admission["status"] == "export_and_admission_verified_training_pending"
            and admission["training_count"] == 1016 and admission["full_registered_denominator"] == 1021
            and admission["protected_development_exclusion_count"] == 48 and admission["source_unchanged"]):
        raise ValueError("The new map has not passed its explicit training-admission contract")
    if not (smoke["status"] == "completed" and smoke["iteration"] == 1100
            and smoke["image_count"] == 1016 and smoke_inputs["input_version"] == "new_map_admission_v1"
            and smoke_config["ply_serialization"] == {"vectorized": True, "native_byte_oracle": True}
            and not smoke_config["gpu_resident_inputs"]):
        raise ValueError("The actual new-map CPU-input training smoke did not satisfy its contract")
    if not recovery.get("full_budget_combination_accepted", False):
        raise ValueError("The full-budget fixed-input recovery combination is not accepted")
    recovery_state = read(args.recovery_acceptance.parent / "STATE.json")
    if not all(recovery_state["checks"].values()):
        raise ValueError("A required full-budget recovery check failed")
    for item in recovery_state["sources"]:
        path = Path(__file__).with_name(Path(item["path"]).name)
        if data_identity(path)["sha256"] != item["sha256"]:
            raise ValueError("The actual full-budget recovery source changed")
    serialized = read(args.smoke / "PLY_SERIALIZATION.json")
    if not serialized["events"] or not all(event["native_byte_oracle"]["identical"] for event in serialized["events"]):
        raise ValueError("The new-map native PLY byte oracle failed")
    for item in smoke_config["implementation"]:
        path = Path(__file__).with_name(Path(item["path"]).name)
        if data_identity(path)["sha256"] != item["sha256"]:
            raise ValueError("The smoke-validated training source changed")
    if len(read(args.localization)["rows"]) != 24 or read(args.localization)["input_version"] != "new_map_admission_v1":
        raise ValueError("The new-map evaluation denominator or version changed")
    image_paths = sorted(path for path in (args.dataset / "images").iterdir()
                         if path.suffix.lower() in (".jpg", ".png", ".jpeg"))
    input_paths = [args.dataset / "sparse/0" / name for name in
                   ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")] + image_paths
    if len(image_paths) != 1016 or [data_identity(path)["sha256"] for path in input_paths] != [
            item["sha256"] for item in smoke_inputs["files"]]:
        raise ValueError("The actual dataset differs from the validated input snapshot")
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise MemoryError("Insufficient free disk for the registered isolated run group")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / "frozen_scripts"
    frozen.mkdir()
    names = ["train_case.py", "runtime.py", "full_intrinsics.py", "checkpoint_adapter.py",
             "gsplat_adapter.py", "ply_serialization.py"]
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), frozen / name)
    state = {"status": "running", "controller_pid": os.getpid(), "runs": [],
        "source": data_identity(Path(__file__)), "frozen_sources": [data_identity(frozen / name) for name in names],
        "input_version": "new_map_admission_v1", "image_count": 1016,
        "admission": data_identity(args.admission), "localization": data_identity(args.localization),
        "smoke": data_identity(args.smoke / "STATE.json"), "recovery_acceptance": data_identity(args.recovery_acceptance),
        "control_iterations": 10000, "candidate_iterations": 12000, "crash_at": 8035,
        "checkpoint_at": 8000, "expected_lost_steps": 35, "native_cpu_images": True,
        "maximum_seconds_per_process": 1800, "watchdog_seconds": 1900,
        "group_disk_budget_bytes": 30 * 1024**3, "new_optimization_count": 0,
        "scope": "new-route compatibility and usability; no aggregate acceleration claim"}
    started = time.monotonic()
    atomic_json(args.output / "STATE.json", state)
    for label, iterations, expected_code in [("reference", 10000, 0), ("interrupted", 12000, 86), ("combined", 12000, 0)]:
        command = [sys.executable, str(frozen / "train_case.py"), "--runtime", str(args.runtime),
            "--dataset", str(args.dataset), "--output", str(args.output / label),
            "--projection", "full", "--seed", "0", "--iterations", str(iterations),
            "--maximum-seconds", "1800", "--input-version", "new_map_admission_v1",
            "--expected-image-count", "1016"]
        if label != "reference":
            command += ["--gsplat-build", str(args.gsplat_build), "--checkpoint-every", "4000",
                        "--vectorized-ply", "--verify-native-ply"]
        if label == "interrupted":
            command += ["--simulate-crash-after", "8035"]
        if label == "combined":
            command += ["--resume-file", str(args.output / "interrupted/checkpoint_008000.pth")]
        row = {"label": label, "command": command, "status": "starting"}
        state["runs"].append(row)
        process_started = time.monotonic()
        with (args.output / (label + ".log")).open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row.update(status="running", pid=process.pid)
            atomic_json(args.output / "STATE.json", state)
            try:
                while process.poll() is None:
                    size = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
                    if time.monotonic() - process_started > 1900 or size > state["group_disk_budget_bytes"]:
                        raise RuntimeError("The registered watchdog or group disk budget was exceeded")
                    time.sleep(2)
            except BaseException as error:
                if process.poll() is None:
                    process.kill()
                process.wait()
                row.update(status="failed_controller", returncode=process.returncode, error=repr(error))
                state["status"] = "failed_controller"
                atomic_json(args.output / "STATE.json", state)
                raise
        row.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - process_started,
                   status="expected_crash" if process.returncode == 86 else "completed" if process.returncode == 0 else "failed")
        if label == "interrupted" and process.returncode == 86:
            child = read(args.output / label / "STATE.json")
            child.update(status="crashed_as_injected", controller_observed_exitcode=86,
                         process_terminated=True, iteration_at_exit=8035)
            atomic_json(args.output / label / "STATE.json", child)
        atomic_json(args.output / "STATE.json", state)
        if process.returncode != expected_code:
            state["status"] = "failed_child"
            atomic_json(args.output / "STATE.json", state)
            raise RuntimeError("A new-route training process violated its registered exit contract")
    reference = read(args.output / "reference/STATE.json")
    combined = read(args.output / "combined/STATE.json")
    restored = read(args.output / "combined/RESTORE_CHECK.json")
    sampling = read(args.output / "combined/SAMPLING_TRACE.json")
    serialized = read(args.output / "combined/PLY_SERIALIZATION.json")
    candidate_counts = dict(Counter(sampling["names"]))
    checks = {
        "reference_completed_10000": reference["status"] == "completed" and reference["iteration"] == 10000,
        "combined_completed_12000": combined["status"] == "completed" and combined["iteration"] == 12000,
        "restored_all_saved_state_exact": restored["status"] == "all_saved_state_exact_before_next_training_iteration"
            and all(restored["checks"].values()),
        "combined_resume_at8000": combined["resumed_iteration"] == 8000 and combined["executed_iterations"] == 4000,
        "combined_total_sampling12000": len(sampling["names"]) == 12000 and sampling["resumed_iteration"] == 8000,
        "all1016_training_cameras_sampled": len(reference["sample_counts"]) == len(candidate_counts) == 1016,
        "reference_total_sampling10000": sum(reference["sample_counts"].values()) == 10000,
        "new_save_schedule_only12000": serialized["schedule_validation"]["expected_process_iterations"]
            == serialized["schedule_validation"]["observed_process_iterations"] == [12000],
        "all_native_ply_bytes_exact": bool(serialized["events"]) and all(event["native_byte_oracle"]["identical"]
            for event in serialized["events"]),
        "no_false_final_model_from_crash": not (args.output / "interrupted/model/point_cloud/iteration_12000/point_cloud.ply").exists(),
    }
    atomic_json(args.output / "SAMPLING_SUMMARY.json", {
        "reference": {"counts": reference["sample_counts"], "mean": 10000 / 1016, "total": 10000},
        "combined": {"counts": candidate_counts, "mean": 12000 / 1016, "total": 12000},
        "discarded_steps_after_checkpoint": 35, "actual_combined_executed_steps_including_redo": 12035})
    state.update(status="training_and_restore_passed_quality_pending" if all(checks.values()) else "failed_checks",
                 checks=checks, elapsed_seconds=time.monotonic() - started,
                 reference_total_process_seconds=state["runs"][0]["elapsed_seconds"],
                 combined_total_process_seconds=sum(row["elapsed_seconds"] for row in state["runs"][1:]))
    atomic_json(args.output / "STATE.json", state)
    print(json.dumps({"status": state["status"], "checks": checks}), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
