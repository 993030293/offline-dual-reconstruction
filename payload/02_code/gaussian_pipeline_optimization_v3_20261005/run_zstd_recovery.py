"""Compare real complete native continuation of the same crash-tested prefix."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("codec-campaign", "shared-prefix-proof", "runtime", "dataset", "build", "camera-config", "localization", "reference-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign, proof = read(args.codec_campaign/"RESULT.json"), read(args.shared_prefix_proof/"RESULT.json")
    archive = args.codec_campaign/"input0_trial0/zstd.archive"
    manifest = read(archive.with_suffix(".archive.json"))
    if not all(campaign["checks"].values()) or not all(proof["checks"].values()) or manifest["native_payload"] != proof["shared_prefix"]:
        raise ValueError("The complete codec or original same-prefix failure proof is missing")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("STATE.json", "CONFIG.json", "INPUT_MANIFEST.json", "CAMERA_MANIFEST.json", "SAMPLING_TRACE.json", "model/point_cloud/iteration_1100/point_cloud.ply"):
        source, target = args.shared_prefix_proof/"resumed"/name, args.output/"native"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if data_identity(source)["sha256"] != data_identity(target)["sha256"]:
            raise ValueError("The exact native continuation reference changed while copying")
    paths = [Path(__file__), Path(__file__).with_name("resume_zstd_camera.py"), Path(__file__).with_name("zstd_checkpoint.py"), args.camera_config]
    sources = [data_identity(path) for path in paths]
    state = {"status": "running", "sources": sources, "runs": [], "shared_prefix": proof["shared_prefix"],
        "control": "actual complete native resume of the same saved500 prefix", "iterations": 1100,
        "all24_previously_viewed_development_frames": True, "PSNR_min_delta_dB": -.1, "SSIM_min_delta": -.002,
        "codec_campaign": str(args.codec_campaign), "new_optimization_count": 0}
    atomic_json(args.output/"PROTOCOL.json", state)

    def run(command, label):
        receipt = args.output/(label+"_process.json")
        monitored = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(receipt),
            "--log", str(args.output/(label+".log")), "--maximum-seconds", "500", "--maximum-rss-gib", "32", "--", *command]
        subprocess.run(monitored, check=True)
        state["runs"].append({"label": label, "process": read(receipt)})
        atomic_json(args.output/"STATE.json", state)

    try:
        run([sys.executable, str(Path(__file__).with_name("resume_zstd_camera.py")), "--checkpoint-archive", str(archive),
            "--transport-output", str(args.output/"transport"), "--camera-config", str(args.camera_config),
            "--runtime", str(args.runtime), "--dataset", str(args.dataset), "--output", str(args.output/"zstd"),
            "--projection", "full", "--seed", "0", "--iterations", "1100", "--maximum-seconds", "1800",
            "--input-version", "new_map_admission_v1", "--expected-image-count", "1016", "--gsplat-build", str(args.build),
            "--vectorized-ply", "--verify-native-ply", "--checkpoint-every", "500"], "resume")
        run([sys.executable, str(V1/"evaluate_pair.py"), "--runtime", str(args.runtime), "--reference-run", str(args.reference_run),
            "--pair", str(args.output), "--control-label", "native", "--candidate-label", "zstd", "--iterations", "1100",
            "--localization", str(args.localization), "--input-version", "new_map_admission_v1"], "evaluation")
        transport = read(args.output/"transport/TRANSPORT.json")
        quality = read(args.output/"evaluation/EVALUATION.json")
        samples = [read(args.output/label/"SAMPLING_TRACE.json")["names"] for label in ("native", "zstd")]
        checks = {"all_three_codec_sources_full_exact_and_lifecycle": all(campaign["checks"].values()),
            "actual_native_model_optimizer_RNG_and_queue_restore": all(transport["restored_state"]["checks"].values()),
            "all1100_samples_exact": samples[0] == samples[1] and len(samples[0]) == 1100,
            "all24_original_quality_guards": quality["development_metric_guard_passed"],
            "verified_owned_expansion_removed_after_complete_restore": transport["expanded_copy_removed"],
            "sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(status="codec_and_actual_recovery_passed_visual_pending" if all(checks.values()) else "failed_verification",
            checks=checks, candidate_minus_control=quality["candidate_minus_control"], transport=transport)
        atomic_json(args.output/"RESULT.json", state)
        if not all(checks.values()):
            raise ValueError("An actual complete continuation or original24 quality guard failed")
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)
    print({"status": state["status"], "checks": checks, "quality_delta": state["candidate_minus_control"]}, flush=True)


if __name__ == "__main__":
    main()
