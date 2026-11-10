"""Validate exact camera combination with fresh paired native training and actual recovery."""

import argparse
import copy
import json
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
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "build", "output", "baseline-config", "candidate-config", "resource-campaign",
                 "combination-oracle", "localization", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    resource = read(args.resource_campaign/"RESULT.json")
    if resource["status"] != "resource_campaign_completed_training_quality_pending" or not resource["sources_unchanged"] or any(
            row["status"] != "resource_passed_training_quality_pending" for row in resource["results"].values()):
        raise ValueError("The three independent camera resource comparisons did not pass")
    oracle = read(args.combination_oracle/"RESULT.json")
    original = read(args.resource_campaign/"trial2/baseline/RESULT.json")
    exact_keys = ("rows", "initial_Gaussians", "before_RNG", "after_RNG", "camera_manifest", "camera_extent")
    if any(oracle[key] != original[key] for key in exact_keys) or not all(oracle["checks"].values()):
        raise ValueError("The combined full Scene differs from the independent original pixel, camera or initialization oracle")
    if shutil.disk_usage(args.output.parent).free < 20*1024**3:
        raise MemoryError("The native camera group has insufficient disk headroom")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), args.baseline_config, args.candidate_config]
    paths += [Path(__file__).with_name(name) for name in ("train_camera_execution.py", "camera_execution.py", "prefetch_camera_images.py")]
    paths += [V1/name for name in ("train_case.py", "checkpoint_adapter.py", "full_intrinsics.py", "gsplat_adapter.py", "evaluate_pair.py")]
    sources = [data_identity(path) for path in paths]
    protocol = {"status": "running", "sources": sources, "runs": [], "seed_checks": [], "frame_count": 1016,
        "all24_previously_viewed_development_views": True, "strict_blind_test": False,
        "controls": "fresh paired native binary reader with Q31 prefetch; camera mechanisms individually off",
        "frozen_old_Q31_quality": "historical; not a current time control; Q52 failure unchanged",
        "PSNR_min_delta_dB": -.1, "SSIM_min_delta": -.002, "seeds": [0, 1, 2],
        "no_metadata_view_or_loss_or_raster_precision_change": True,
        "individual_resource_gains_from": str(args.resource_campaign), "combination_oracle": str(args.combination_oracle),
        "maximum_group_seconds": 3800, "maximum_group_disk_bytes": 30*1024**3,
        "recovery": "1100 continuous, actual exit86 at600, resume complete500 checkpoint", "counted": False}
    state = copy.deepcopy(protocol)
    atomic_json(args.output/"PROTOCOL.json", protocol)
    started = time.monotonic()

    def run(command, label, seconds, expected=0):
        if sources != [data_identity(path) for path in paths] or time.monotonic()-started >= 3800:
            raise ValueError("The frozen native camera group changed its source or exceeded its time bound")
        receipt = args.output/(label+"_process.json")
        bounded = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(receipt),
            "--log", str(args.output/(label+".log")), "--maximum-seconds", str(seconds), "--maximum-rss-gib", "32", "--", *command]
        row = {"label": label, "status": "running", "command": bounded}
        state["runs"].append(row)
        atomic_json(args.output/"STATE.json", state)
        with (args.output/(label+"_controller.log")).open("x") as stream:
            result = subprocess.call(bounded, stdout=stream, stderr=subprocess.STDOUT)
        observed = read(receipt)
        row.update(returncode=observed["returncode"], elapsed_seconds=observed["elapsed_seconds"],
            status="expected_crash" if expected == 86 and observed["returncode"] == 86 else "completed" if result == 0 else "failed")
        atomic_json(args.output/"STATE.json", state)
        if observed["returncode"] != expected or expected == 0 and result != 0:
            raise RuntimeError("A native camera training or evaluation did not match its declared exit")
        if sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) > 30*1024**3:
            raise MemoryError("The native camera group exceeded its declared disk budget")

    def training(output, seed, iterations, candidate):
        return [sys.executable, str(Path(__file__).with_name("train_camera_execution.py")),
            "--camera-config", str(args.candidate_config if candidate else args.baseline_config),
            "--runtime", str(args.runtime), "--dataset", str(args.dataset), "--output", str(output), "--projection", "full",
            "--seed", str(seed), "--iterations", str(iterations), "--maximum-seconds", "1800",
            "--input-version", "new_map_admission_v1", "--expected-image-count", "1016",
            "--gsplat-build", str(args.build), "--vectorized-ply", "--verify-native-ply"]

    def evaluate(pair, control, candidate, iterations, label):
        run([sys.executable, str(V1/"evaluate_pair.py"), "--runtime", str(args.runtime), "--reference-run", str(args.reference_run),
            "--pair", str(pair), "--control-label", control, "--candidate-label", candidate, "--iterations", str(iterations),
            "--localization", str(args.localization), "--input-version", "new_map_admission_v1"], label, 300)
        result = read(pair/"evaluation/EVALUATION.json")
        row = {"label": label, "all24_quality_guards_passed": result["development_metric_guard_passed"],
            "candidate_minus_control": result["candidate_minus_control"], "summary": {name: result["results"][name]["summary"] for name in (control, candidate)}}
        # A failed seed remains in the explicit denominator before the controller raises.
        state["seed_checks"].append(row)
        atomic_json(args.output/"STATE.json", state)
        if not result["development_metric_guard_passed"]:
            raise ValueError("An unchanged all24 PSNR or SSIM protection failed")
        return row

    try:
        recovery = args.output/"recovery"
        for name in ("continuous", "interrupted", "resumed"):
            command = training(recovery/name, 0, 1100, True)+["--checkpoint-every", "500"]
            if name == "interrupted":
                command += ["--simulate-crash-after", "600"]
            elif name == "resumed":
                command += ["--resume-file", str(recovery/"interrupted/checkpoint_000500.pth")]
            run(command, "recovery_"+name, 500, 86 if name == "interrupted" else 0)
            if name == "interrupted":
                native_state = read(recovery/name/"STATE.json")
                native_state.update(status="crashed_as_injected", controller_observed_exitcode=86, process_terminated=True)
                atomic_json(recovery/name/"STATE.json", native_state)
        trace = [read(recovery/name/"SAMPLING_TRACE.json")["names"] for name in ("continuous", "resumed")]
        restore = read(recovery/"resumed/RESTORE_CHECK.json")
        checks = {"all1100_samples_exact": trace[0] == trace[1] and len(trace[0]) == 1100,
            "all_saved_tensor_optimizer_RNG_queue_state_restored_exact": all(restore["checks"].values()),
            "actual_crash_produced_no_final_model": not (recovery/"interrupted/model/point_cloud/iteration_1100/point_cloud.ply").exists(),
            "resumed_iteration500": read(recovery/"resumed/STATE.json")["resumed_iteration"] == 500}
        checkpoint = recovery/"interrupted/checkpoint_000500.pth"
        identity = read(checkpoint.with_suffix(".json"))["identity"]
        for name in ("opaque_alpha", "clampfree", "calibrated_projection", "source_sha256"):
            invalid = copy.deepcopy(identity)
            if name == "source_sha256":
                invalid["external_adapters"][name][0] = "0"*64
            else:
                flags = invalid["external_adapters"]["camera_execution"]
                flags[name] = not flags[name]
            try:
                RecoveryState(recovery, invalid, 500, checkpoint).prepare(None, SimpleNamespace(iterations=1100), None)
                checks[name+"_mismatch_rejected"] = False
            except ValueError as error:
                checks[name+"_mismatch_rejected"] = "identity mismatch" in str(error)
        state["recovery_checks"] = checks
        if not all(checks.values()):
            raise ValueError("An actual complete camera recovery or mechanism identity protection failed")
        evaluate(recovery, "continuous", "resumed", 1100, "recovery_evaluation")
        for seed in range(3):
            pair = args.output/f"seed{seed}"
            for name in (("baseline", "combined") if seed % 2 == 0 else ("combined", "baseline")):
                run(training(pair/name, seed, 12000, name == "combined"), f"seed{seed}_"+name, 1900)
            states = [read(pair/name/"STATE.json") for name in ("baseline", "combined")]
            manifests = [read(pair/name/"INPUT_MANIFEST.json") for name in ("baseline", "combined")]
            exact = states[0]["sample_sequence_sha256"] == states[1]["sample_sequence_sha256"] and [
                row["sha256"] for row in manifests[0]["files"]] == [row["sha256"] for row in manifests[1]["files"]]
            if not exact:
                raise ValueError("An actual12000 camera sample sequence or frozen input payload changed")
            row = evaluate(pair, "baseline", "combined", 12000, f"seed{seed}_evaluation")
            row.update(seed=seed, input_and_all12000_samples_exact=True,
                actual_training_seconds={name: value["training_call_seconds"] for name, value in zip(("baseline", "combined"), states)},
                actual_peak_RSS_bytes={name: value["maximum_rss_bytes"] for name, value in zip(("baseline", "combined"), states)})
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(row), flush=True)
        state.update(status="three_fresh_seed_quality_and_actual_recovery_passed_visual_pending", sources_unchanged=True,
            elapsed_seconds=time.monotonic()-started, counted=False)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
