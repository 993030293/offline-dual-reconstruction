"""Run full continuous and resumed training plus byte-exact checkpoint transport checks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from run_side import windows_to_wsl
from build_raw_viewer import read_vertices


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--case", default="box2")
    args = parser.parse_args()
    side = args.campaign / args.case / "optimized"
    output = args.campaign / args.case / "checkpoint_modules"
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    admitted = read(side / "export/TRAINING_ADMISSION.json")
    common = ["--mode", "optimized", "--runtime", windows_to_wsl(ROOT / "04_outputs/box2_3dgs_display_v1_20260930"),
        "--dataset", windows_to_wsl(Path(admitted["dataset"])), "--projection", "full", "--seed", "0",
        "--iterations", "12000", "--maximum-seconds", "1800", "--input-version", args.case+"_current_raw_recovery_audit",
        "--expected-image-count", str(admitted["training_count"]), "--gsplat-build",
        windows_to_wsl(ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p6_gsplat_build_001"),
        "--vectorized-ply", "--verify-native-ply", "--checkpoint-every", "6000"]
    state = {"status": "running", "rows": [], "source": data_identity(Path(__file__)),
        "primary_side_outputs_unchanged": True, "all_training_images_used": admitted["training_count"],
        "all_trials_are_full12000_schedule_not_short_smoke": True}
    atomic_json(output / "STATE.json", state)
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def run(label, arguments, budget=4350):
        command = ["wsl.exe", "-d", "E26-C2-LIV-20260907-134802", "--", "env", "PYTHONUTF8=1",
            "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "/opt/e26_thin_surface_v1/venv/bin/python",
            windows_to_wsl(CODE / "checkpoint_variant.py"), *common, "--output", windows_to_wsl(output / label), *arguments]
        lease = [sys.executable, CODE / "gpu_lease.py", "--receipt", output / (label+"_GPU_LEASE.json"),
                 "--label", "checkpoint_audit_"+label, "--", *command]
        guarded = [sys.executable, V1 / "run_bounded_task.py", "--receipt", output / (label+"_process.json"),
                   "--log", output / (label+".log"), "--maximum-seconds", str(budget), "--maximum-rss-gib", "8", "--", *lease]
        before = time.perf_counter()
        result = subprocess.run(list(map(str, guarded)), env=environment)
        measured = read(output / (label+"_process.json"))
        state["rows"].append({"label": label, "returncode": result.returncode, "wall_seconds": time.perf_counter()-before,
                              "process": measured})
        atomic_json(output / "STATE.json", state)
        if result.returncode:
            raise RuntimeError("Full checkpoint training trial failed: "+label)
        trained = read(output / label / "STATE.json")
        if trained["status"] != "completed" or trained["iteration"] != 12000:
            raise ValueError("A training schedule is incomplete")

    try:
        run("continuous", ["--restore-mechanism", "native"])
        checkpoint = output / "continuous/checkpoint_006000.pth"
        if read(checkpoint.with_suffix(".json"))["status"] != "complete":
            raise ValueError("No complete native checkpoint was saved")
        for mechanism in ("native", "combined"):
            run("resume_"+mechanism, ["--restore-mechanism", mechanism, "--resume-file", windows_to_wsl(checkpoint)])
        import numpy as np
        names = read(output / "continuous/SAMPLING_TRACE.json")["names"]
        original = read_vertices(output / "continuous/model/point_cloud/iteration_12000/point_cloud.ply")[1]
        checks = {}
        for label in ("resume_native", "resume_combined"):
            sample = read(output / label / "SAMPLING_TRACE.json")
            restored = read(output / label / "RESTORE_CHECK.json")
            current = read_vertices(output / label / "model/point_cloud/iteration_12000/point_cloud.ply")[1]
            equal_population = len(original) == len(current) and original.dtype == current.dtype
            differences = {key: float(np.max(np.abs(original[key]-current[key]))) for key in original.dtype.names} if equal_population else None
            checks[label] = {"all12000_camera_names_exact": sample["names"] == names and len(names) == 12000,
                "nine_complete_saved_state_and_rng_checks_exact": len(restored["checks"]) == 9 and all(restored["checks"].values()),
                "final_PLY_equal_population": equal_population, "all_attributes_close": equal_population and all(np.allclose(original[key], current[key], atol=1e-6, rtol=1e-5) for key in original.dtype.names),
                "complete_attribute_max_abs_differences": differences}
        result = {**state, "status": "full_training_recovery_trials_complete", "checks": checks,
            "complete_saved_state_and_sampler_passed": all(row["all12000_camera_names_exact"] and row["nine_complete_saved_state_and_rng_checks_exact"] for row in checks.values()),
            "final_training_numeric_diagnostic_passed": all(row["all_attributes_close"] for row in checks.values()),
            "elapsed_seconds": time.perf_counter()-started, "checkpoint": data_identity(checkpoint)}
        atomic_json(output / "RESULT.json", result)
    except BaseException as error:
        state.update(status="failed_audit_attempt", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
