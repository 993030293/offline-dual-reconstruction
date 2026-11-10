"""Run prespecified native CPU and GPU input-residency training pairs serially."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "output", "build", "validation", "smoke"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--training-source-dir", type=Path)
    parser.add_argument("--previous-pairs", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    root = args.training_source_dir or Path(__file__).parent
    validation = read(args.validation)
    if (validation["status"] != "input_and_resource_guards_passed_training_pending"
            or validation["rss_reduction"] < .2 or len(validation["pairs"]) != 3
            or not all(validation["numeric_checks"].values())
            or not all(p["all212_values_layout_and_geometry_exact"] for p in validation["pairs"])):
        raise ValueError("Input equivalence and resource admission did not pass")
    for row in validation["sources"]:
        name = Path(row["path"]).name
        if name != "validate_input_residency.py" and data_identity(root / name)["sha256"] != row["sha256"]:
            raise ValueError("A validated shared input implementation changed")
    smoke = read(args.smoke / "STATE.json")
    smoke_config = read(args.smoke / "CONFIG.json")
    residency = smoke["scene_timing"]["input_residency"]
    if (smoke["status"] != "completed" or smoke["iteration"] != 1100 or smoke["image_count"] != 212
            or smoke["gaussian_count"] <= 35633 or not smoke_config["gpu_resident_inputs"]
            or not smoke_config["trim_host_after_load"] or residency["camera_count"] != 212
            or residency["payload_bytes_by_device"]["cpu"] != 0
            or residency["payload_bytes_by_device"]["cuda"] != 1707668480):
        raise ValueError("The actual native-residency growth smoke did not pass")
    for row in smoke_config["implementation"]:
        if data_identity(root / Path(row["path"]).name)["sha256"] != row["sha256"]:
            raise ValueError("Training sources differ from the actual smoke")
    if len(args.previous_pairs) != args.seed or (args.seed and args.training_source_dir is None):
        raise ValueError("Later fixed seeds require every earlier accepted pair and frozen training sources")
    for seed, previous in enumerate(args.previous_pairs):
        prior = read(previous / "PAIR_STATE.json")
        evaluation = read(previous / "evaluation/EVALUATION.json")
        visual = read(previous / "evaluation/VISUAL_REVIEW.json")
        if (prior["seed"] != seed or prior["status"] != "training_completed_pending_evaluation"
                or evaluation["control_label"] != "reference" or evaluation["candidate_label"] != "gpu_resident"
                or not evaluation["development_metric_guard_passed"]
                or evaluation["all_frames_included_in_denominator"] != 24
                or not visual["passed"] or visual["independent_review"]):
            raise ValueError("An earlier fixed seed failed or lacks actual visual review")
        for row in prior["sources"]:
            if data_identity(root / Path(row["path"]).name)["sha256"] != row["sha256"]:
                raise ValueError("All fixed seeds must use identical source content")
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3:
        raise MemoryError("Insufficient free disk")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "frozen_scripts"
    snapshot.mkdir()
    names = ["train_case.py", "runtime.py", "full_intrinsics.py", "gsplat_adapter.py", "host_memory.py"]
    for name in names:
        shutil.copyfile(root / name, snapshot / name)
    state = {"status": "running", "candidate": "P26_INPUT_RESIDENCY", "seed": args.seed,
        "controller_pid": os.getpid(), "iterations": 12000, "runs": [], "counted": False,
        "validation": data_identity(args.validation), "smoke": data_identity(args.smoke / "STATE.json"),
        "sources": [data_identity(snapshot / name) for name in names], "source": data_identity(Path(__file__)),
        "equal_trim_in_both_variants": True, "maximum_median_time_ratio": 1.1,
        "gpu_concurrency": 1, "maximum_seconds_each": 1800}
    atomic_json(args.output / "PAIR_STATE.json", state)
    started = time.monotonic()
    order = ["gpu_resident", "reference"] if args.seed == 1 else ["reference", "gpu_resident"]
    try:
        for label in order:
            command = [sys.executable, str(snapshot / "train_case.py"), "--runtime", str(args.runtime),
                "--dataset", str(args.dataset), "--output", str(args.output / label), "--projection", "full",
                "--seed", str(args.seed), "--iterations", "12000", "--gsplat-build", str(args.build),
                "--trim-host-after-load"]
            if label == "gpu_resident":
                command.append("--gpu-resident-inputs")
            row = {"label": label, "command": command, "status": "starting"}
            state["runs"].append(row)
            with (args.output / (label + ".log")).open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                row.update(pid=process.pid, status="running")
                atomic_json(args.output / "PAIR_STATE.json", state)
                try:
                    code = process.wait(timeout=1900)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    row["status"] = "watchdog_timeout"
                    raise
            row.update(returncode=code, status="completed" if code == 0 else "failed")
            atomic_json(args.output / "PAIR_STATE.json", state)
            if code:
                raise RuntimeError("A residency training process failed")
        states = {label: read(args.output / label / "STATE.json") for label in order}
        inputs = [read(args.output / label / "INPUT_MANIFEST.json")["files"] for label in order]
        if ([r["sha256"] for r in inputs[0]] != [r["sha256"] for r in inputs[1]]
                or states[order[0]]["sample_sequence_sha256"] != states[order[1]]["sample_sequence_sha256"]):
            raise ValueError("Paired input or sampling identities differ")
        times = {label: states[label]["training_call_seconds"] for label in order}
        state.update(status="training_completed_pending_evaluation", identical_input_and_sampling=True,
            training_seconds=times, time_ratio=times["gpu_resident"] / times["reference"],
            elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output / "PAIR_STATE.json", state)
        raise
    atomic_json(args.output / "PAIR_STATE.json", state)
    print(json.dumps({key: state[key] for key in ("status", "seed", "training_seconds", "time_ratio")}))


if __name__ == "__main__":
    main()
