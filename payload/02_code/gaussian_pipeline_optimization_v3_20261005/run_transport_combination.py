"""Validate three actual native versus combined same-prefix continuations."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("codec-campaign", "shared-prefix-proof", "runtime", "dataset", "build", "camera-config", "localization", "reference-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    campaign, proof = read(args.codec_campaign/"RESULT.json"), read(args.shared_prefix_proof/"RESULT.json")
    archive = args.codec_campaign/"input0_trial0/zstd.archive"
    manifest = read(archive.with_suffix(".archive.json"))
    prefix = Path(proof["shared_prefix"]["path"])
    if not all(campaign["checks"].values()) or not all(proof["checks"].values()) or manifest["native_payload"] != data_identity(prefix):
        raise ValueError("The accepted complete transport and common original prefix are missing")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("resume_transport_camera.py"), Path(__file__).with_name("zstd_checkpoint.py"),
        Path(__file__).with_name("private_checkpoint_load.py"), args.camera_config, prefix, archive]
    identities = [data_identity(path) for path in paths]
    state = {"status": "running", "sources_inputs": identities, "runs": [], "pairs": [], "same_fixed_native500_prefix": proof["shared_prefix"],
        "three_actual500_to1100_pairs": True, "complete_state_and_all1100_samples_and_all24_original_quality": True,
        "PSNR_min_delta_dB": -.1, "SSIM_min_delta": -.002, "new_optimization_count": 0,
        "maximum_group_seconds": 900, "expanded_native_copies_preserved": True,
        "previously_materialized_archive_encode_not_in_restore_cost": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()

    def run(command, label, seconds):
        if identities != [data_identity(path) for path in paths] or time.perf_counter()-started > 900:
            raise ValueError("A combined recovery source or bounded schedule changed")
        receipt = args.output/(label+"_process.json")
        row = {"label": label, "status": "running", "command": command}
        state["runs"].append(row)
        atomic_json(args.output/"STATE.json", state)
        subprocess.run([sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(receipt),
            "--log", str(args.output/(label+".log")), "--maximum-seconds", str(seconds), "--maximum-rss-gib", "32", "--", *command], check=True)
        row.update(status="completed", process=read(receipt))
        atomic_json(args.output/"STATE.json", state)

    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            for mode in (("native", "zstd_mapped") if trial % 2 == 0 else ("zstd_mapped", "native")):
                run([sys.executable, str(Path(__file__).with_name("resume_transport_camera.py")), "--transport-mode", mode,
                    "--native-checkpoint", str(prefix), "--checkpoint-archive", str(archive), "--transport-output", str(pair/(mode+"_transport")),
                    "--camera-config", str(args.camera_config), "--runtime", str(args.runtime), "--dataset", str(args.dataset),
                    "--output", str(pair/mode), "--projection", "full", "--seed", "0", "--iterations", "1100", "--maximum-seconds", "1800",
                    "--input-version", "new_map_admission_v1", "--expected-image-count", "1016", "--gsplat-build", str(args.build),
                    "--vectorized-ply", "--verify-native-ply", "--checkpoint-every", "500"], f"trial{trial}_"+mode, 180)
            run([sys.executable, str(V1/"evaluate_pair.py"), "--runtime", str(args.runtime), "--reference-run", str(args.reference_run),
                "--pair", str(pair), "--control-label", "native", "--candidate-label", "zstd_mapped", "--iterations", "1100",
                "--localization", str(args.localization), "--input-version", "new_map_admission_v1"], f"trial{trial}_evaluation", 90)
            transports = {mode: read(pair/(mode+"_transport")/"TRANSPORT.json") for mode in ("native", "zstd_mapped")}
            states = {mode: read(pair/mode/"STATE.json") for mode in transports}
            samples = [read(pair/mode/"SAMPLING_TRACE.json")["names"] for mode in transports]
            quality = read(pair/"evaluation/EVALUATION.json")
            checks = {"all_native_complete_restores": all(all(t["restored_state"]["checks"].values()) for t in transports.values()),
                "same_saved_model_optimizer_RNG_camera_sampling": samples[0] == samples[1] and len(samples[0]) == 1100,
                "all24_original_quality": quality["development_metric_guard_passed"],
                "original_archive_private_copies_and_sources_preserved": all(t["all_sources_and_payloads_unchanged"] for t in transports.values()),
                "two_original_and_two_private_mapped_device_preserving_loads": all(len(t["loader"]["calls"]) == 2
                    and all(c["mmap"] == (mode == "zstd_mapped") and c["map_location_argument_unchanged"] == "None" for c in t["loader"]["calls"])
                    for mode, t in transports.items()),
                "global_loader_restored": all(t["loader"]["original_global_loader_restored"] for t in transports.values())}
            row = {"trial": trial, "checks": checks, "candidate_minus_control": quality["candidate_minus_control"],
                "complete_prepare_verify_seconds": {mode: t["complete_prepare_and_verify_seconds"] for mode, t in transports.items()},
                "complete_native_load_seconds": {mode: sum(c["seconds"] for c in t["loader"]["calls"]) for mode, t in transports.items()},
                "actual_training_call_seconds": {mode: states[mode]["training_call_seconds"] for mode in transports}}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(row), flush=True)
            if not all(checks.values()):
                raise ValueError("A related transport/loading combination failed an original protection")
        state.update(status="three_actual_transport_private_load_combinations_passed",
            medians={key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("native", "zstd_mapped")}
                for key in ("complete_prepare_verify_seconds", "complete_native_load_seconds", "actual_training_call_seconds")},
            sources_inputs_unchanged=identities == [data_identity(path) for path in paths], elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
