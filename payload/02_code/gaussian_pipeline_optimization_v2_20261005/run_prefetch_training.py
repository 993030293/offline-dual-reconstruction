"""Run the preregistered three-seed training and development-quality protection."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "build", "validation", "smoke", "localization", "reference-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    resource = read(args.validation)
    smoke = read(args.smoke/"STATE.json")
    adapter = read(args.smoke/"PREFETCH_ADAPTER.json")
    if resource["status"] != "input_resource_guards_passed_training_pending" or not all(resource["checks"].values()):
        raise ValueError("The three-pair resource admission did not pass")
    if (smoke["status"] != "completed" or smoke["iteration"] != 1100 or smoke["image_count"] != 1016
            or smoke["gaussian_count"] <= 75767 or adapter["mode"] != "prefetch"
            or adapter["prefetch"][0]["consumed"] != 1016):
        raise ValueError("The actual1016-input growth smoke is incomplete")
    if any(data_identity(Path(s["path"])) != s for s in adapter["sources"]):
        raise ValueError("The actual smoke used a different training adapter")
    if shutil.disk_usage(args.output.parent).free < 20*1024**3:
        raise MemoryError("Insufficient free disk for bounded paired training")
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path(__file__).with_name("train_with_camera_prefetch.py"),
        Path(__file__).with_name("prefetch_camera_images.py"), V1/"train_case.py",
        V1/"full_intrinsics.py", V1/"gsplat_adapter.py", V1/"ply_serialization.py", V1/"evaluate_pair.py"]
    sources = [data_identity(path) for path in source_paths]
    snapshot = args.output/"source_snapshots"
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot/path.name)
    state = {"status": "running", "controller_pid": os.getpid(), "mechanism": "Q31_BOUNDED_TRAINING_IMAGE_PREFETCH",
        "input_version": "new_map_admission_v1", "image_count": 1016, "seeds": [0, 1, 2],
        "iterations_per_run": 12000, "same_backend": "gsplat1.5.3", "sources": sources,
        "resources": data_identity(args.validation), "smoke": data_identity(args.smoke/"STATE.json"),
        "localization": data_identity(args.localization), "all_development_views": 24,
        "minimum_PSNR_delta_dB": -.1, "minimum_SSIM_delta": -.002,
        "maximum_training_time_ratio": 1.1, "maximum_seconds_per_training": 1800,
        "maximum_RSS_bytes": 32*1024**3, "maximum_group_disk_bytes": 30*1024**3,
        "runs": [], "seed_checks": [], "counted": False, "visual_review_pending": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    atomic_json(args.output/"STATE.json", state)
    started = time.monotonic()
    def run(command, name, seconds):
        if sources != [data_identity(path) for path in source_paths]:
            raise ValueError("A registered training source changed between runs")
        bounded = [sys.executable, str(V1/"run_bounded_task.py"),
            "--receipt", str(args.output/(name+"_process.json")), "--log", str(args.output/(name+".log")),
            "--maximum-seconds", str(seconds), "--maximum-rss-gib", "32", "--", *command]
        with (args.output/(name+"_controller.log")).open("x") as log:
            process = subprocess.Popen(bounded, stdout=log, stderr=subprocess.STDOUT)
            row = {"name": name, "status": "running", "pid": process.pid, "command": bounded}
            state["runs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            code = process.wait()
        row.update(status="completed" if code == 0 else "failed", returncode=code)
        state["elapsed_seconds"] = time.monotonic()-started
        atomic_json(args.output/"STATE.json", state)
        if code:
            raise RuntimeError("A bounded prefetch training or evaluation process failed")
    try:
        for seed in range(3):
            pair = args.output/f"seed{seed}"
            pair.mkdir()
            order = ("prefetch", "reference") if seed == 1 else ("reference", "prefetch")
            for label in order:
                command = [sys.executable, str(Path(__file__).with_name("train_with_camera_prefetch.py")),
                    "--prefetch-mode", "baseline" if label == "reference" else "prefetch",
                    "--resource-validation", str(args.validation), "--runtime", str(args.runtime),
                    "--dataset", str(args.dataset), "--output", str(pair/label), "--projection", "full",
                    "--seed", str(seed), "--iterations", "12000", "--maximum-seconds", "1800",
                    "--input-version", "new_map_admission_v1", "--expected-image-count", "1016",
                    "--gsplat-build", str(args.build), "--vectorized-ply", "--verify-native-ply"]
                run(command, f"seed{seed}_{label}", 1900)
            control, candidate = [read(pair/label/"STATE.json") for label in ("reference", "prefetch")]
            manifests = [read(pair/label/"INPUT_MANIFEST.json") for label in ("reference", "prefetch")]
            identical = (control["sample_sequence_sha256"] == candidate["sample_sequence_sha256"] and
                [r["sha256"] for r in manifests[0]["files"]] == [r["sha256"] for r in manifests[1]["files"]])
            if not identical:
                raise ValueError("Actual training input or complete sample sequence changed")
            command = [sys.executable, str(V1/"evaluate_pair.py"), "--runtime", str(args.runtime),
                "--reference-run", str(args.reference_run), "--pair", str(pair), "--control-label", "reference",
                "--candidate-label", "prefetch", "--iterations", "12000", "--localization", str(args.localization),
                "--input-version", "new_map_admission_v1"]
            run(command, f"seed{seed}_evaluation", 300)
            evaluation = read(pair/"evaluation/EVALUATION.json")
            ratio = candidate["training_call_seconds"]/control["training_call_seconds"]
            row = {"seed": seed, "all24_metric_guards_passed": evaluation["development_metric_guard_passed"],
                "input_and_complete_sampling_exact": identical, "training_time_ratio": ratio,
                "training_time_protection_passed": ratio <= 1.1,
                "candidate_minus_control": evaluation["candidate_minus_control"],
                "reference_training_seconds": control["training_call_seconds"],
                "candidate_training_seconds": candidate["training_call_seconds"], "visual_review_pending": True}
            state["seed_checks"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(row), flush=True)
            if not row["all24_metric_guards_passed"] or not row["training_time_protection_passed"]:
                raise ValueError("A registered seed failed; do not replace it or continue selecting seeds")
        state.update(status="three_seed_training_and_metrics_passed_visual_and_combination_pending",
                     elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed_training_or_metric_guard", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
