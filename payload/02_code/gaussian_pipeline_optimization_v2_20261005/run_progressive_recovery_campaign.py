"""Measure complete native recovery pairs without sharing a mutable database."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from assess_progressive_recovery import compare_recovery

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "input-images", "initial-frames", "preflight", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    diagnostic = read(args.preflight/"RESULT.json")
    if diagnostic["status"] != "recovery_geometry_protected" or not all(diagnostic["checks"].values()):
        raise ValueError("Actual first recovery must satisfy the unchanged geometry policy")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("run_progressive_pair_recovery.py"),
        Path(__file__).with_name("assess_progressive_recovery.py"), Path(__file__).with_name("native_observation_checks.py"),
        V1/"run_pair_recovery.py", V1/"run_sfm_preflight.py",
        ROOT/"05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/config/METRIC_ALIGNMENT_POLICY.json",
        ROOT/"05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/code/validated_fresh_runtime/metric_admission.py"]
    identities = [data_identity(p) for p in sources]
    snapshot = args.output/"sources"
    snapshot.mkdir()
    for p in sources:
        shutil.copyfile(p, snapshot/p.name)
    state = {"status": "running", "sources": identities, "runs": [], "pairs": [], "counted": False,
        "minimum_complete_recovery_reduction": .1, "minimum_original_observation_retention_ratio": 1.,
        "no_baseline_registered_or_admitted_name_loss": True, "mean_reprojection_error_max_delta_px": .05,
        "all1002_denominator_and_all_frame_guard_unchanged": True,
        "new_total_observations_disclosed_separately": True, "three_actual_pairs": True,
        "maximum_seconds_each": 1260, "maximum_group_seconds": 3800, "maximum_RSS_bytes": 32*1024**3,
        "metric_BA_training_and24_view_quality_remain_pending": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    atomic_json(args.output/"STATE.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["baseline", "progressive"] if trial % 2 == 0 else ["progressive", "baseline"]):
                if time.monotonic()-started > 3800 or identities != [data_identity(p) for p in sources]:
                    raise ValueError("The recovery group budget or frozen implementation changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "1260", "--maximum-rss-gib", "32", "--",
                    sys.executable, str(Path(__file__).with_name("run_progressive_pair_recovery.py")),
                    "--mode", mode, "--output", str(pair/mode)]
                for name in ("source_run", "input_images", "initial_frames"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                row = {"trial": trial, "mode": mode, "status": "running"}
                state["runs"].append(row)
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("An actual recovery pair failed")
            geometry = compare_recovery(pair/"baseline/RESULT.json", pair/"progressive/RESULT.json")
            atomic_json(pair/"GEOMETRY.json", geometry)
            results = {m: read(pair/m/"RESULT.json") for m in ("baseline", "progressive")}
            item = {"trial": trial, "all_geometry_guards": all(geometry["checks"].values()),
                "complete_recovery_seconds": {m: r["elapsed_seconds_including_preparation"] for m, r in results.items()},
                "registered": {m: r["final"]["registered_count"] for m, r in results.items()},
                "admitted": geometry["admitted"], "requested_pairs": {m: r["requested_pairs"] for m, r in results.items()},
                "total_observations": geometry["total_observations"],
                "retained_original_observations": geometry["original_observations_retained"]}
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(item), flush=True)
            if not item["all_geometry_guards"]:
                raise ValueError("A complete recovery pair lost protected camera geometry")
        median = {m: statistics.median(p["complete_recovery_seconds"][m] for p in state["pairs"]) for m in ("baseline", "progressive")}
        reduction = 1-median["progressive"]/median["baseline"]
        result = {"status": "three_recovery_pairs_passed_metric_training_quality_pending" if reduction >= .1 else "failed_speed_guard",
            "pairs": state["pairs"], "median_complete_seconds": median, "complete_recovery_reduction": reduction,
            "counted": False, "stage_quality_pending": True, "full_route_combination_accepted": False}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
