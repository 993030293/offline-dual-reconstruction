"""Run the registered fixed-seed target-density quality comparison serially."""

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
    for name in ("runtime", "dataset", "output", "build", "validation", "smoke", "roi-source"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--training-source-dir", type=Path)
    parser.add_argument("--seed-zero-evaluation", type=Path)
    args = parser.parse_args()
    source_root = args.training_source_dir or Path(__file__).parent
    validation = json.loads(args.validation.read_text())
    smoke = json.loads((args.smoke/"STATE.json").read_text())
    smoke_config = json.loads((args.smoke/"CONFIG.json").read_text())
    if validation["status"] != "passed" or not all(validation["checks"].values()):
        raise ValueError("Target density validation failed")
    if (data_identity(source_root/"target_density_adapter.py")["sha256"] != validation["adapter"]["sha256"]
            or data_identity(args.roi_source)["sha256"] != validation["roi_source"]["sha256"]
            or data_identity(args.dataset/"sparse/0/points3D.ply")["sha256"] != validation["ROI_rows"][0]["source"]["sha256"]):
        raise ValueError("Target ROI, adapter or sparse initialization differs from validation")
    density = smoke.get("target_density_summary")
    if (smoke["status"] != "completed" or smoke["iteration"] != 1100 or smoke["image_count"] != 212
            or not density or density["extra_candidates_inside_sum"] <= 0
            or density["extra_candidates_outside_sum"] != 0):
        raise ValueError("Actual target-only growth smoke did not pass")
    for row in smoke_config["implementation"]:
        if data_identity(source_root/Path(row["path"]).name)["sha256"] != row["sha256"]:
            raise ValueError("Source differs from the actual smoke")
    if args.seed != 0:
        if args.seed_zero_evaluation is None or args.training_source_dir is None:
            raise ValueError("Later seeds require the first pair's frozen source and quality receipt")
        evaluation = json.loads(args.seed_zero_evaluation.read_text())
        if (not evaluation["development_metric_guard_passed"]
                or evaluation["candidate_minus_control"]["target"]["mean_PSNR_dB"] < .15
                or evaluation["control_label"] != "reference" or evaluation["candidate_label"] != "target_density"):
            raise ValueError("The first target-quality pair failed admission")
        first_pair = json.loads((args.seed_zero_evaluation.parent.parent/"PAIR_STATE.json").read_text())
        if not first_pair["resource_ratio_guard_passed"]:
            raise ValueError("The first pair exceeded the registered time or Gaussian-count ratio")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"frozen_scripts"
    snapshot.mkdir()
    names = ["train_case.py", "runtime.py", "full_intrinsics.py", "gsplat_adapter.py", "target_density_adapter.py"]
    for name in names:
        shutil.copyfile(source_root/name, snapshot/name)
    shutil.copyfile(args.roi_source, args.output/"FROZEN_ROI.json")
    state = {"status": "running", "controller_pid": os.getpid(), "seed": args.seed, "runs": [],
        "candidate": "P17_TARGET_DENSITY", "iterations": 12000, "multiplier": 2., "dataset": str(args.dataset),
        "gpu_concurrency": 1, "maximum_seconds_each": 1800, "maximum_time_ratio": 1.25,
        "maximum_final_gaussian_ratio": 1.5, "target_PSNR_minimum_per_seed_dB": .15,
        "target_PSNR_minimum_three_seed_median_dB": .2,
        "validation": data_identity(args.validation), "smoke": data_identity(args.smoke/"STATE.json"),
        "build": data_identity(args.build/"RESULT.json"), "roi": data_identity(args.output/"FROZEN_ROI.json"),
        "source": data_identity(Path(__file__)), "sources": [data_identity(snapshot/name) for name in names],
        "counted": False, "evaluation_role": "previously_viewed_development"}
    atomic_json(args.output/"PAIR_STATE.json", state)
    started = time.monotonic()
    for label in (["target_density", "reference"] if args.seed == 1 else ["reference", "target_density"]):
        command = [sys.executable, str(snapshot/"train_case.py"), "--runtime", str(args.runtime),
            "--dataset", str(args.dataset), "--output", str(args.output/label), "--projection", "full",
            "--seed", str(args.seed), "--iterations", "12000", "--gsplat-build", str(args.build)]
        if label == "target_density":
            command.extend(["--target-density-roi", str(args.output/"FROZEN_ROI.json")])
        row = {"label": label, "command": command, "status": "starting"}
        state["runs"].append(row)
        with (args.output/(label+".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row.update(pid=process.pid, status="running")
            atomic_json(args.output/"PAIR_STATE.json", state)
            try:
                code = process.wait(timeout=1900)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                row["status"] = "watchdog_timeout"
                state["status"] = "failed"
                atomic_json(args.output/"PAIR_STATE.json", state)
                raise
        row.update(returncode=code, status="completed" if code == 0 else "failed")
        atomic_json(args.output/"PAIR_STATE.json", state)
        if code:
            state["status"] = "failed"
            atomic_json(args.output/"PAIR_STATE.json", state)
            raise RuntimeError("Target density training failed")
    control, candidate = [json.loads((args.output/label/"STATE.json").read_text()) for label in ("reference", "target_density")]
    controls, candidates = [json.loads((args.output/label/"INPUT_MANIFEST.json").read_text())["files"] for label in ("reference", "target_density")]
    if ([row["sha256"] for row in controls] != [row["sha256"] for row in candidates]
            or control["sample_sequence_sha256"] != candidate["sample_sequence_sha256"]):
        state["status"] = "failed_input_or_sampling_sequence"
        atomic_json(args.output/"PAIR_STATE.json", state)
        raise ValueError("The frozen input or sampling contract changed")
    time_ratio = candidate["training_call_seconds"]/control["training_call_seconds"]
    count_ratio = candidate["gaussian_count"]/control["gaussian_count"]
    state.update(status="training_completed_pending_evaluation", elapsed_seconds=time.monotonic()-started,
        identical_input_and_sampling=True, reference_seconds=control["training_call_seconds"],
        candidate_seconds=candidate["training_call_seconds"], time_ratio=time_ratio, gaussian_count_ratio=count_ratio,
        resource_ratio_guard_passed=time_ratio <= 1.25 and count_ratio <= 1.5)
    atomic_json(args.output/"PAIR_STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "seed", "reference_seconds", "candidate_seconds", "resource_ratio_guard_passed")}))


if __name__ == "__main__":
    main()
