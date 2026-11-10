"""Validate actual recovery and three native training seeds against frozen Q31 controls."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from checkpoint_adapter import RecoveryState


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "build", "output", "adapter-config", "controls", "localization", "reference-run", "resource-validation"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    resource = read(args.resource_validation)
    if resource["status"] != "resource_guards_passed_training_combination_pending" or not resource["sources_unchanged"]:
        raise ValueError("The real full Scene resource campaign did not pass")
    if shutil.disk_usage(args.output.parent).free < 20*1024**3:
        raise MemoryError("The bounded native training group has insufficient disk")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), args.adapter_config, Path(__file__).with_name("train_optimized_case.py"),
        Path(__file__).with_name("prefetch_camera_images.py"), Path(__file__).with_name("training_camera_metadata.py")]
    paths += [V1/name for name in ("train_case.py", "checkpoint_adapter.py", "gsplat_adapter.py", "full_intrinsics.py", "evaluate_pair.py")]
    sources = [data_identity(p) for p in paths]
    frozen_controls = []
    for seed in range(3):
        source = args.controls/f"seed{seed}/prefetch"
        config, state = read(source/"CONFIG.json"), read(source/"STATE.json")
        if (state["status"] != "completed" or state["iteration"] != 12000 or state["image_count"] != 1016
                or config["seed"] != seed or config["input_version"] != "new_map_admission_v1"
                or config["intervention"]["projection"] != "full"):
            raise ValueError("A frozen Q31 control does not match the declared full training protocol")
        files = ["STATE.json", "CONFIG.json", "INPUT_MANIFEST.json", "CAMERA_MANIFEST.json",
                 "model/point_cloud/iteration_12000/point_cloud.ply"]
        target = args.output/f"seed{seed}/reference"
        target.mkdir(parents=True)
        identities = []
        for name in files:
            identity = data_identity(source/name)
            (target/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target/name)
            if identity["sha256"] != data_identity(target/name)["sha256"]:
                raise ValueError("A readonly frozen control copy changed its payload")
            identities.append(identity)
        frozen_controls.append({"seed": seed, "source": str(source), "original_identities": identities,
            "copied_to": str(target), "used_for_quality_only": True, "not_current_speed_evidence": True})
    state = {"status": "running", "pid": os.getpid(), "runs": [], "seed_checks": [], "sources": sources,
        "frozen_control_bindings": frozen_controls, "input_version": "new_map_admission_v1", "frame_count": 1016,
        "all24_views": True, "development_previously_viewed": True, "PSNR_min_delta_dB": -.1, "SSIM_min_delta": -.002,
        "recovery_budget": {"iterations": 1100, "checkpoint": 500, "crash_after": 600},
        "maximum_group_seconds": 2400, "maximum_group_disk_bytes": 30*1024**3,
        "historical_training_times_not_compared": True, "new_optimization_count": 0,
        "Q52_reader_resource_gain_separate": resource["reader_stage_reduction"], "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()

    def run(command, label, seconds, expected=0):
        if sources != [data_identity(p) for p in paths] or time.monotonic()-started > 2400:
            raise ValueError("A frozen source or native campaign time budget changed")
        bounded = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output/(label+"_process.json")),
            "--log", str(args.output/(label+".log")), "--maximum-seconds", str(seconds), "--maximum-rss-gib", "32", "--", *command]
        with (args.output/(label+"_controller.log")).open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(bounded, stdout=stream, stderr=subprocess.STDOUT)
            row = {"label": label, "pid": child.pid, "status": "running", "command": bounded}
            state["runs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            code = child.wait()
        receipt = read(args.output/(label+"_process.json"))
        row.update(controller_returncode=code, actual_returncode=receipt["returncode"],
            status="expected_crash" if expected == 86 and receipt["returncode"] == 86 else "completed" if code == 0 else "failed")
        atomic_json(args.output/"STATE.json", state)
        if receipt["returncode"] != expected or (expected == 0 and code != 0):
            raise RuntimeError("A real native training process did not match its declared exit")
        if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > 30*1024**3:
            raise MemoryError("The native validation group disk budget was exceeded")

    def training(output, seed, iterations):
        return [sys.executable, str(Path(__file__).with_name("train_optimized_case.py")),
            "--adapter-config", str(args.adapter_config), "--runtime", str(args.runtime), "--dataset", str(args.dataset),
            "--output", str(output), "--projection", "full", "--seed", str(seed), "--iterations", str(iterations),
            "--maximum-seconds", "1800", "--input-version", "new_map_admission_v1", "--expected-image-count", "1016",
            "--gsplat-build", str(args.build), "--vectorized-ply", "--verify-native-ply"]

    def evaluation(pair, control, candidate, iterations, label):
        run([sys.executable, str(V1/"evaluate_pair.py"), "--runtime", str(args.runtime), "--reference-run", str(args.reference_run),
            "--pair", str(pair), "--control-label", control, "--candidate-label", candidate, "--iterations", str(iterations),
            "--localization", str(args.localization), "--input-version", "new_map_admission_v1"], label, 300)
        result = read(pair/"evaluation/EVALUATION.json")
        if not result["development_metric_guard_passed"]:
            raise ValueError("An unchanged complete 24-view development guard failed")
        return result

    try:
        recovery = args.output/"recovery"
        recovery.mkdir()
        for label in ("continuous", "interrupted", "resumed"):
            command = training(recovery/label, 0, 1100)+["--checkpoint-every", "500"]
            if label == "interrupted":
                command += ["--simulate-crash-after", "600"]
            if label == "resumed":
                command += ["--resume-file", str(recovery/"interrupted/checkpoint_000500.pth")]
            run(command, "recovery_"+label, 500, 86 if label == "interrupted" else 0)
            if label == "interrupted":
                child_state = read(recovery/label/"STATE.json")
                child_state.update(status="crashed_as_injected", controller_observed_exitcode=86, process_terminated=True)
                atomic_json(recovery/label/"STATE.json", child_state)
        traces = {label: read(recovery/label/"SAMPLING_TRACE.json")["names"] for label in ("continuous", "resumed")}
        restore = read(recovery/"resumed/RESTORE_CHECK.json")
        checks = {"all1100_camera_samples_exact": traces["continuous"] == traces["resumed"] and len(traces["continuous"]) == 1100,
            "all_saved_tensor_optimizer_RNG_and_queue_state_exact": restore["status"] == "all_saved_state_exact_before_next_training_iteration" and all(restore["checks"].values()),
            "actual_exit86_no_final_model": not (recovery/"interrupted/model/point_cloud/iteration_1100/point_cloud.ply").exists(),
            "resumed_from500": read(recovery/"resumed/STATE.json")["resumed_iteration"] == 500}
        checkpoint = recovery/"interrupted/checkpoint_000500.pth"
        identity = read(checkpoint.with_suffix(".json"))["identity"]
        for field in ("configuration", "source_sha256"):
            invalid = copy.deepcopy(identity)
            if field == "configuration":
                invalid["external_adapters"][field]["training_metadata"]["enabled"] = False
            else:
                invalid["external_adapters"][field][0] = "0"*64
            try:
                RecoveryState(recovery, invalid, 500, checkpoint).prepare(None, SimpleNamespace(iterations=1100), None)
                checks["external_"+field+"_mismatch_rejected"] = False
            except ValueError as error:
                checks["external_"+field+"_mismatch_rejected"] = "identity mismatch" in str(error)
        if not all(checks.values()):
            raise ValueError("The real crash, continuation or adapter identity guard failed")
        recovery_eval = evaluation(recovery, "continuous", "resumed", 1100, "recovery_evaluation")
        state["recovery"] = {"checks": checks, "candidate_minus_control": recovery_eval["candidate_minus_control"],
            "visual_review_pending": True, "new_optimization_count": 0}
        atomic_json(args.output/"STATE.json", state)
        for seed in range(3):
            pair = args.output/f"seed{seed}"
            run(training(pair/"metadata", seed, 12000), f"seed{seed}_metadata", 1900)
            control, candidate = [read(pair/label/"STATE.json") for label in ("reference", "metadata")]
            manifests = [read(pair/label/"INPUT_MANIFEST.json") for label in ("reference", "metadata")]
            exact = control["sample_sequence_sha256"] == candidate["sample_sequence_sha256"] and [r["sha256"] for r in manifests[0]["files"]] == [r["sha256"] for r in manifests[1]["files"]]
            if not exact:
                raise ValueError("The complete native training input or random camera sequence changed")
            result = evaluation(pair, "reference", "metadata", 12000, f"seed{seed}_evaluation")
            row = {"seed": seed, "input_and_all12000_samples_exact": exact,
                "all24_quality_guards_passed": result["development_metric_guard_passed"],
                "candidate_minus_control": result["candidate_minus_control"], "visual_review_pending": True,
                "historical_training_time_not_used_for_speed_claim": True}
            state["seed_checks"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(row), flush=True)
        state.update(status="three_seed_quality_and_actual_recovery_passed_visual_combination_pending", elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
