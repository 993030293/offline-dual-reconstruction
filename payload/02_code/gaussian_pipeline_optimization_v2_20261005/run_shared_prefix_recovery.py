"""Test actual recovery against uninterrupted continuation of the identical saved prefix."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "runtime", "dataset", "build", "camera-config", "localization", "reference-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.source_run/"checkpoint_000500.pth"
    source_state, checkpoint_receipt = read(args.source_run/"STATE.json"), read(checkpoint.with_suffix(".json"))
    config = read(args.source_run/"CONFIG.json")
    if source_state["status"] != "completed" or source_state["iteration"] != 1100 or config["parameters"]["densify_until_iter"] != 7000:
        raise ValueError("A same-prefix uninterrupted source does not match the original recovery budget")
    if checkpoint_receipt["checkpoint"]["sha256"] != data_identity(checkpoint)["sha256"]:
        raise ValueError("The same-prefix complete checkpoint changed its identity")
    for name in ("STATE.json", "CONFIG.json", "INPUT_MANIFEST.json", "CAMERA_MANIFEST.json", "SAMPLING_TRACE.json",
                 "model/point_cloud/iteration_1100/point_cloud.ply"):
        target = args.output/"continuous"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source_run/name, target)
        if data_identity(target)["sha256"] != data_identity(args.source_run/name)["sha256"]:
            raise ValueError("A same-prefix uninterrupted reference copy changed")
    paths = [Path(__file__), args.camera_config]+[Path(__file__).with_name(name) for name in
        ("train_camera_execution.py", "camera_execution.py", "prefetch_camera_images.py")]
    sources = [data_identity(path) for path in paths]
    state = {"status": "running", "sources": sources, "shared_prefix": data_identity(checkpoint), "runs": [],
        "uninterrupted_source": str(args.source_run), "actual_crash_after": 600, "resume_iteration": 500, "iterations": 1100,
        "PSNR_min_delta_dB": -.1, "SSIM_min_delta": -.002, "all24_development_views": True,
        "failed_independent_prefix_comparison_preserved": "camera_execution_training_001",
        "diagnostic_reason": "independent prefixes already differ at500 before crash; this test holds the saved prefix fixed",
        "new_optimization_count": 0, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)

    def run(command, label, expected=0):
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("The same-prefix source identity changed")
        receipt = args.output/(label+"_process.json")
        bounded = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(receipt),
            "--log", str(args.output/(label+".log")), "--maximum-seconds", "500", "--maximum-rss-gib", "32", "--", *command]
        with (args.output/(label+"_controller.log")).open("x") as stream:
            code = subprocess.call(bounded, stdout=stream, stderr=subprocess.STDOUT)
        process = read(receipt)
        state["runs"].append({"label": label, "process": process})
        atomic_json(args.output/"STATE.json", state)
        if process["returncode"] != expected or expected == 0 and code != 0:
            raise RuntimeError("A same-prefix native process failed its declared exit")

    try:
        for name in ("interrupted", "resumed"):
            command = [sys.executable, str(Path(__file__).with_name("train_camera_execution.py")),
                "--camera-config", str(args.camera_config), "--runtime", str(args.runtime), "--dataset", str(args.dataset),
                "--output", str(args.output/name), "--projection", "full", "--seed", "0", "--iterations", "1100",
                "--maximum-seconds", "1800", "--input-version", "new_map_admission_v1", "--expected-image-count", "1016",
                "--gsplat-build", str(args.build), "--vectorized-ply", "--verify-native-ply",
                "--checkpoint-every", "500", "--resume-file", str(checkpoint)]
            if name == "interrupted":
                command += ["--simulate-crash-after", "600"]
            run(command, name, 86 if name == "interrupted" else 0)
            if name == "interrupted":
                child = read(args.output/name/"STATE.json")
                child.update(status="crashed_as_injected", controller_observed_exitcode=86, process_terminated=True)
                atomic_json(args.output/name/"STATE.json", child)
        restore = {name: read(args.output/name/"RESTORE_CHECK.json") for name in ("interrupted", "resumed")}
        samples = [read(args.output/name/"SAMPLING_TRACE.json")["names"] for name in ("continuous", "resumed")]
        checks = {"identical_actual_complete_prefix_restored_in_both_branches": all(all(row["checks"].values()) for row in restore.values()),
            "all1100_samples_match_uninterrupted_same_prefix": samples[0] == samples[1] and len(samples[0]) == 1100,
            "crashed_branch_has_no_final_model": not (args.output/"interrupted/model/point_cloud/iteration_1100/point_cloud.ply").exists(),
            "source_complete_checkpoint_preserved": state["shared_prefix"] == data_identity(checkpoint)}
        if not all(checks.values()):
            raise AssertionError("A true same-prefix recovery state or sampling oracle failed")
        run([sys.executable, str(V1/"evaluate_pair.py"), "--runtime", str(args.runtime), "--reference-run", str(args.reference_run),
            "--pair", str(args.output), "--control-label", "continuous", "--candidate-label", "resumed", "--iterations", "1100",
            "--localization", str(args.localization), "--input-version", "new_map_admission_v1"], "evaluation")
        evaluation = read(args.output/"evaluation/EVALUATION.json")
        checks["same_prefix_all24_original_PSNR_and_SSIM_guard"] = evaluation["development_metric_guard_passed"]
        state.update(checks=checks, candidate_minus_control=evaluation["candidate_minus_control"],
            status="same_prefix_recovery_passed_visual_pending" if all(checks.values()) else "failed_original_quality_guard")
        atomic_json(args.output/"RESULT.json", state)
        if not all(checks.values()):
            raise AssertionError("The same-prefix recovery still failed its unchanged quality guard")
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)
    print({key: state[key] for key in ("status", "checks", "candidate_minus_control")}, flush=True)


if __name__ == "__main__":
    main()
