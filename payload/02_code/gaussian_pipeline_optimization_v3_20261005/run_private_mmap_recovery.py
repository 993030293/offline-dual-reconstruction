"""Compare an actual mapped continuation with the preserved same-prefix native run."""

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
    for name in ("campaign", "shared-prefix-proof", "runtime", "dataset", "build", "camera-config", "localization", "reference-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign, proof = read(args.campaign/"RESULT.json"), read(args.shared_prefix_proof/"RESULT.json")
    if not all(campaign["checks"].values()) or not all(proof["checks"].values()):
        raise ValueError("The corrected complete loader profile or original same-prefix proof is missing")
    source = Path(proof["shared_prefix"]["path"])
    if data_identity(source) != proof["shared_prefix"]:
        raise ValueError("The fixed native500 prefix changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("STATE.json", "CONFIG.json", "INPUT_MANIFEST.json", "CAMERA_MANIFEST.json", "SAMPLING_TRACE.json", "model/point_cloud/iteration_1100/point_cloud.ply"):
        original, target = args.shared_prefix_proof/"resumed"/name, args.output/"native"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if data_identity(original)["sha256"] != data_identity(target)["sha256"]:
            raise ValueError("The original actual native continuation reference changed while copying")
    paths = [Path(__file__), Path(__file__).with_name("resume_private_mmap_camera.py"),
        Path(__file__).with_name("private_checkpoint_load.py"), args.camera_config]
    sources = [data_identity(p) for p in paths]
    state = {"status": "running", "sources": sources, "runs": [], "shared_prefix": proof["shared_prefix"],
        "actual_saved500_to1100_default_device_native_continuation": True,
        "PSNR_min_delta_dB": -.1, "SSIM_min_delta": -.002, "all24_required": True, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)

    def run(command, label):
        receipt = args.output/(label+"_process.json")
        subprocess.run([sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(receipt), "--log", str(args.output/(label+".log")),
            "--maximum-seconds", "500", "--maximum-rss-gib", "32", "--", *command], check=True)
        state["runs"].append({"label": label, "process": read(receipt)})
        atomic_json(args.output/"STATE.json", state)

    try:
        run([sys.executable, str(Path(__file__).with_name("resume_private_mmap_camera.py")), "--native-checkpoint", str(source),
            "--transport-output", str(args.output/"transport"), "--camera-config", str(args.camera_config),
            "--runtime", str(args.runtime), "--dataset", str(args.dataset), "--output", str(args.output/"mapped"),
            "--projection", "full", "--seed", "0", "--iterations", "1100", "--maximum-seconds", "1800",
            "--input-version", "new_map_admission_v1", "--expected-image-count", "1016", "--gsplat-build", str(args.build),
            "--vectorized-ply", "--verify-native-ply", "--checkpoint-every", "500"], "resume")
        run([sys.executable, str(V1/"evaluate_pair.py"), "--runtime", str(args.runtime), "--reference-run", str(args.reference_run),
            "--pair", str(args.output), "--control-label", "native", "--candidate-label", "mapped", "--iterations", "1100",
            "--localization", str(args.localization), "--input-version", "new_map_admission_v1"], "evaluation")
        loader = read(args.output/"transport/LOADER.json")
        quality = read(args.output/"evaluation/EVALUATION.json")
        samples = [read(args.output/label/"SAMPLING_TRACE.json")["names"] for label in ("native", "mapped")]
        checks = {"corrected_three_source_three_pair_cost_guards": all(campaign["checks"].values()),
            "all_original_tensor_optimizer_RNG_camera_restore_guards": all(loader["restored_state"]["checks"].values()),
            "all1100_samples_exact": samples[0] == samples[1] and len(samples[0]) == 1100,
            "all24_original_quality_guards": quality["development_metric_guard_passed"],
            "exactly_two_owned_default_device_private_loads": len(loader["trace"]["calls"]) == 2
                and all(c["mmap"] and c["map_location_argument_unchanged"] == "None" for c in loader["trace"]["calls"]),
            "original_loader_restored_and_owned_source_preserved_unchanged": loader["trace"]["original_global_loader_restored"]
                and loader["owned_copy_preserved"] and loader["owned_source_and_original_unchanged"],
            "sources_unchanged": sources == [data_identity(p) for p in paths]}
        state.update(status="native_private_mapped_recovery_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
            checks=checks, loader=loader, candidate_minus_control=quality["candidate_minus_control"])
        atomic_json(args.output/"RESULT.json", state)
        if not all(checks.values()):
            raise ValueError("A complete private mapped recovery or original quality guard failed")
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)
    print({"status": state["status"], "checks": checks, "quality_delta": state["candidate_minus_control"]}, flush=True)


if __name__ == "__main__":
    main()
