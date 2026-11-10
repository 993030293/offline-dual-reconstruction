"""Compare exact compact host inputs with native storage using frozen training sources."""

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
    for name in ("runtime", "dataset", "output", "build", "validation", "smoke"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--training-source-dir", type=Path)
    parser.add_argument("--seed-zero-evaluation", type=Path)
    args = parser.parse_args()
    source_root = args.training_source_dir or Path(__file__).parent
    numeric = json.loads(args.validation.read_text())
    if (numeric["status"] != "input_and_memory_guards_passed_training_pending"
            or not numeric["equal_trim_in_both_variants"]
            or not all(numeric["numeric_checks"].values()) or numeric["rss_reduction"] < .2
            or len(numeric["pairs"]) != 3
            or any(not row["all212_inputs_and_geometry_exact"]
                   or row["compact"]["storage"]["payload_reduction"] < .7
                   for row in numeric["pairs"])):
        raise ValueError("Actual three-pair input and memory admission failed")
    if data_identity(source_root/"host_image_storage.py")["sha256"] != numeric["adapter"]["sha256"]:
        raise ValueError("The numerically validated storage implementation changed")
    smoke = json.loads((args.smoke/"STATE.json").read_text())
    smoke_config = json.loads((args.smoke/"CONFIG.json").read_text())
    host = smoke["host_image_storage"]
    if (smoke["status"] != "completed" or smoke["iteration"] != 1100 or smoke["image_count"] != 212
            or smoke["gaussian_count"] <= 35633 or not smoke_config["compact_host_images"]
            or not smoke_config["trim_host_after_load"]
            or host["samples"] != 1100 or host["camera_count"] != 212
            or host["active_camera"] is not None or not host["only_active_camera_has_float_inputs"]):
        raise ValueError("Actual host-storage growth and lifecycle smoke failed")
    for row in smoke_config["implementation"]:
        if data_identity(source_root/Path(row["path"]).name)["sha256"] != row["sha256"]:
            raise ValueError("Training sources differ from the actual smoke")
    if args.seed != 0:
        if args.seed_zero_evaluation is None or args.training_source_dir is None:
            raise ValueError("Later seeds require first-pair frozen sources and evaluation")
        evaluation = json.loads(args.seed_zero_evaluation.read_text())
        first = json.loads((args.seed_zero_evaluation.parent.parent/"PAIR_STATE.json").read_text())
        if (not evaluation["development_metric_guard_passed"] or evaluation["control_label"] != "reference"
                or evaluation["candidate_label"] != "compact_host" or first["time_ratio"] > 1.1):
            raise ValueError("First-pair quality or memory-mode cost admission failed")
    if shutil.disk_usage(args.output.parent).free < 20*1024**3:
        raise RuntimeError("Insufficient free disk for an isolated pair")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"frozen_scripts"
    snapshot.mkdir()
    names = ["train_case.py", "runtime.py", "full_intrinsics.py", "gsplat_adapter.py",
             "host_image_storage.py", "host_memory.py"]
    for name in names:
        shutil.copyfile(source_root/name, snapshot/name)
    state = {"status": "running", "controller_pid": os.getpid(), "seed": args.seed, "runs": [],
        "candidate": "P24_HOST_IMAGE_STORAGE", "iterations": 12000, "dataset": str(args.dataset),
        "gpu_concurrency": 1, "maximum_seconds_each": 1800, "projection": "full", "ssim": "reference",
        "equal_trim_in_both_variants": True, "maximum_median_training_time_ratio": 1.1,
        "validation": data_identity(args.validation), "smoke": data_identity(args.smoke/"STATE.json"),
        "build": data_identity(args.build/"RESULT.json"), "source": data_identity(Path(__file__)),
        "sources": [data_identity(snapshot/name) for name in names], "counted": False,
        "maximum_optimization_count_if_accepted": 1}
    atomic_json(args.output/"PAIR_STATE.json", state)
    started = time.monotonic()
    order = ["compact_host", "reference"] if args.seed == 1 else ["reference", "compact_host"]
    for label in order:
        command = [sys.executable, str(snapshot/"train_case.py"), "--runtime", str(args.runtime),
            "--dataset", str(args.dataset), "--output", str(args.output/label), "--projection", "full",
            "--seed", str(args.seed), "--iterations", "12000", "--gsplat-build", str(args.build),
            "--trim-host-after-load"]
        if label == "compact_host":
            command.append("--compact-host-images")
        row = {"label": label, "command": command, "status": "starting"}
        state["runs"].append(row)
        with (args.output/(label + ".log")).open("w") as log:
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
        if code:
            state["status"] = "failed"
        atomic_json(args.output/"PAIR_STATE.json", state)
        if code:
            raise RuntimeError("Host-storage paired training failed")
    control, candidate = [json.loads((args.output/label/"STATE.json").read_text())
                          for label in ("reference", "compact_host")]
    inputs = [json.loads((args.output/label/"INPUT_MANIFEST.json").read_text())["files"]
              for label in ("reference", "compact_host")]
    if ([row["sha256"] for row in inputs[0]] != [row["sha256"] for row in inputs[1]]
            or control["sample_sequence_sha256"] != candidate["sample_sequence_sha256"]):
        state["status"] = "failed_input_or_sampling_sequence"
        atomic_json(args.output/"PAIR_STATE.json", state)
        raise ValueError("Frozen inputs or camera sample sequence changed")
    state.update(status="training_completed_pending_evaluation", elapsed_seconds=time.monotonic()-started,
        reference_seconds=control["training_call_seconds"], candidate_seconds=candidate["training_call_seconds"],
        identical_input_and_sampling=True, time_ratio=candidate["training_call_seconds"]/control["training_call_seconds"])
    atomic_json(args.output/"PAIR_STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "seed", "reference_seconds", "candidate_seconds", "time_ratio")}))


if __name__ == "__main__":
    main()
