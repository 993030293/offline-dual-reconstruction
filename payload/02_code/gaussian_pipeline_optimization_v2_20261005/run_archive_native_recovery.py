"""Require a complete real native resume and all development quality guards for compressed state."""

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
    for name in ("archive-campaign", "shared-prefix-proof", "runtime", "dataset", "build", "camera-config",
                 "localization", "reference-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    archive_result = read(args.archive_campaign/"RESULT.json")
    proof = read(args.shared_prefix_proof/"RESULT.json")
    archive = args.archive_campaign/"trial0.ptgz"
    manifest = read(archive.with_suffix(".ptgz.json"))
    if not all(archive_result["checks"].values()) or not all(proof["checks"].values()):
        raise ValueError("The real complete-state archive and actual identical-prefix crash proof must pass first")
    if manifest["native_payload"] != proof["shared_prefix"]:
        raise ValueError("The archive does not contain the exact actual crash-tested prefix")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("STATE.json", "CONFIG.json", "INPUT_MANIFEST.json", "CAMERA_MANIFEST.json", "SAMPLING_TRACE.json",
                 "model/point_cloud/iteration_1100/point_cloud.ply"):
        source, target = args.shared_prefix_proof/"resumed"/name, args.output/"native"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if data_identity(source)["sha256"] != data_identity(target)["sha256"]:
            raise ValueError("The actual native recovery reference changed while copying")
    paths = [Path(__file__), Path(__file__).with_name("resume_archived_camera.py"),
             Path(__file__).with_name("checkpoint_archive.py"), args.camera_config]
    sources = [data_identity(path) for path in paths]
    state = {"status": "running", "sources": sources, "runs": [], "archive_resource_campaign": str(args.archive_campaign),
        "real_same_prefix_crash_proof": str(args.shared_prefix_proof), "shared_prefix": proof["shared_prefix"],
        "control": "actual uncompressed native resume of the same checkpoint", "iterations": 1100,
        "all24_previously_viewed_development_views": True, "PSNR_min_delta_dB": -.1, "SSIM_min_delta": -.002,
        "new_optimization_count": 0, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)

    def run(command, label):
        receipt = args.output/(label+"_process.json")
        bounded = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(receipt),
            "--log", str(args.output/(label+".log")), "--maximum-seconds", "500", "--maximum-rss-gib", "32", "--", *command]
        with (args.output/(label+"_controller.log")).open("x") as stream:
            code = subprocess.call(bounded, stdout=stream, stderr=subprocess.STDOUT)
        process = read(receipt)
        state["runs"].append({"label": label, "process": process})
        atomic_json(args.output/"STATE.json", state)
        if code or process["returncode"]:
            raise RuntimeError("The actual archived recovery or evaluation failed")

    try:
        run([sys.executable, str(Path(__file__).with_name("resume_archived_camera.py")),
            "--checkpoint-archive", str(archive), "--transport-output", str(args.output/"transport"),
            "--camera-config", str(args.camera_config), "--runtime", str(args.runtime), "--dataset", str(args.dataset),
            "--output", str(args.output/"archived"), "--projection", "full", "--seed", "0", "--iterations", "1100",
            "--maximum-seconds", "1800", "--input-version", "new_map_admission_v1", "--expected-image-count", "1016",
            "--gsplat-build", str(args.build), "--vectorized-ply", "--verify-native-ply", "--checkpoint-every", "500"], "resume")
        run([sys.executable, str(V1/"evaluate_pair.py"), "--runtime", str(args.runtime), "--reference-run", str(args.reference_run),
            "--pair", str(args.output), "--control-label", "native", "--candidate-label", "archived", "--iterations", "1100",
            "--localization", str(args.localization), "--input-version", "new_map_admission_v1"], "evaluation")
        transport = read(args.output/"transport/TRANSPORT.json")
        quality = read(args.output/"evaluation/EVALUATION.json")
        samples = [read(args.output/name/"SAMPLING_TRACE.json")["names"] for name in ("native", "archived")]
        checks = {"three_actual_archive_resource_and_payload_oracles_passed": all(archive_result["checks"].values()),
            "actual_native_model_optimizer_RNG_and_camera_queue_restored": all(transport["restored_state"]["checks"].values()),
            "all1100_sample_names_exact": samples[0] == samples[1] and len(samples[0]) == 1100,
            "all24_original_quality_protection": quality["development_metric_guard_passed"],
            "recoverable_expansion_removed_after_success": transport["expanded_copy_removed"],
            "transport_sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(checks=checks, candidate_minus_control=quality["candidate_minus_control"], transport=transport,
            status="resource_and_native_recovery_passed_visual_pending" if all(checks.values()) else "failed_verification")
        atomic_json(args.output/"RESULT.json", state)
        if not all(checks.values()):
            raise AssertionError("An actual archived complete-state recovery protection failed")
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)
    print({"status": state["status"], "checks": state["checks"], "candidate_minus_control": state["candidate_minus_control"]}, flush=True)


if __name__ == "__main__":
    main()
