"""Copy the current complete model inputs and launch one bounded original evaluator profile."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/evaluation_workspace_001"
MODELS = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/camera_quality_training_001/seed0"
LOC = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p27_newmap_localization_lazy_serial_001/H_LOCALIZATION.json"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def linux(path):
    return "/mnt/c/" + Path(path).as_posix()[3:]


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", choices=("d0", "d1"), default="d0")
    parser.add_argument("--cuda-events", action="store_true")
    args = parser.parse_args()
    output = OUT / args.diagnostic
    output.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    locality = json.loads(LOC.read_text())
    if locality["input_version"] != "new_map_admission_v1" or len(locality["rows"]) != 24:
        raise ValueError("The full current localization denominator changed")
    inputs = [LOC]
    for row in locality["rows"]:
        name = row["image_reference"].replace("\\", "/")
        inputs.append(Path("C:/" + name[7:]) if name.startswith("/mnt/c/") else Path(name))
    for label in ("baseline", "combined"):
        state = json.loads((MODELS / label / "STATE.json").read_text())
        manifest = json.loads((MODELS / label / "INPUT_MANIFEST.json").read_text())
        if state["status"] != "completed" or state["iteration"] != 12000 or manifest["input_version"] != "new_map_admission_v1" or manifest["image_count"] != 1016:
            raise ValueError("The current actual model or full training denominator changed")
        inputs.extend(MODELS / label / name for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"))
    identities = [data_identity(path) for path in inputs]
    sources = [data_identity(path) for path in [Path(__file__), CODE / "profile_evaluation_workspaces.py", CODE / "evaluate_view_major.py", CODE / "evaluate_interior_ssim.py", CODE / "view_major_evaluation.py"]]
    copies = output / "source_copies"
    copies.mkdir()
    for i, row in enumerate(sources):
        shutil.copyfile(row["path"], copies / (f"{i:02d}_" + Path(row["path"]).name))
    for label in ("baseline", "combined"):
        for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"):
            original, copied = MODELS / label / name, output / "pair" / label / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, copied)
            if data_identity(original)["sha256"] != data_identity(copied)["sha256"]:
                raise ValueError("A complete copied evaluation input changed")
    atomic_json(output / "CONTRACT.json", {"sources": sources, "inputs": identities, "complete24": True, "models12000_members1016": True,
                "profiling_only_no_optimization_acceptance": True, "setup_seconds": time.perf_counter() - started})
    command = [sys.executable, str(CODE / "run_bounded_journal_task.py"), "--receipt", str(output / "WINDOWS_GUARD.json"),
               "--log", str(output / "PROFILE.log"), "--maximum-seconds", "300", "--maximum-rss-gib", "8", "--",
               "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
               linux(CODE / "profile_evaluation_workspaces.py"), "--pair", linux(output / "pair"), "--localization", linux(LOC)]
    if args.cuda_events:
        command.append("--cuda-events")
    subprocess.run(command, check=True)
    profile = json.loads((output / "PROFILE.json").read_text())
    if profile["status"] != "complete_native_profile_passed" or not all(profile["checks"].values()):
        raise ValueError("The actual complete original evaluator profile did not pass")
    if identities != [data_identity(path) for path in inputs] or sources != [data_identity(row["path"]) for row in sources]:
        raise ValueError("The complete profile changed its original source or inputs")
    atomic_json(output / "RESULT.json", {"status": "actual_complete_native_evaluation_profile_protected", "sources": sources,
                "inputs": identities, "complete_entry_seconds": time.perf_counter() - started, "new_optimization_count": 0})
    print(json.dumps({"status": "actual_complete_native_evaluation_profile_protected", "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
