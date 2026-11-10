"""Compare three cold complete localization pairs without relaxing original guards."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWNED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001"
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    generic = read(OWNED / "g3/RESULT.json")
    if generic["status"] != "passed" or not all(generic["checks"].values()):
        raise ValueError("The current complete pixel and real error/cleanup oracle did not pass")
    paths = [Path(__file__), CODE / "localize_gpu_camera_remap.py", CODE / "camera_pixel_gpu_remap.py",
             CODE / "camera_remap_stdio_worker.py", CODE / "camera_remap_matching_guard.py",
             CODE / "gpu_camera_remap_pixels.py", CODE / "camera_pixel_remap.py", CODE / "mapped_image_linux_guard.py",
             ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/compare_actual_localization.py"]
    sources = [data_identity(path) for path in paths]
    frozen = {row["path"]: row for row in generic["sources"]}
    if any(row != frozen[row["path"]] for row in sources if row["path"] in frozen):
        raise ValueError("The actual pixel implementation differs from its current complete oracle")
    saved = args.output / "source_snapshot"
    saved.mkdir()
    for ordinal, path in enumerate(paths):
        shutil.copyfile(path, saved / (f"{ordinal:03d}_" + path.name))
    state = {"status": "running", "sources": sources, "runs": [], "pairs": [],
             "all_original24_references_48_matches_and_protected_denominators": True,
             "cold_private_reference_and_feature_caches_every_side": True,
             "minimum_complete_stage_reduction": .05, "maximum_each_whole_ratio": 1.1,
             "whole_median_cannot_regress": True,
             "all_prior_negative_first_failed_and_required_review_costs_shared_six_sides_before_acceptance": True,
             "speed_results_pending_final_full_cost": True, "new_optimization_count": 0}
    state["registered_early_stop"] = "stop_after_first_pair_if_both_complete_phase_and_whole_are_not_faster_no_acceptance"
    atomic_json(args.output / "PROTOCOL.json", state)
    comparisons = 0.
    try:
        for trial, order in enumerate((("native", "CUDA"), ("CUDA", "native"), ("native", "CUDA"))):
            folder = args.output / f"t{trial}"
            folder.mkdir()
            results, whole = {}, {}
            for mode in order:
                if time.perf_counter() - STARTED > 1700 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The frozen complete source or bounded schedule changed")
                command = [sys.executable, "-B", str(CODE / "run_bounded_journal_task.py"),
                    "--receipt", str(folder / (mode + "_GUARD.json")), "--log", str(folder / (mode + ".log")),
                    "--maximum-seconds", "450", "--maximum-rss-gib", "8", "--", sys.executable, "-B",
                    str(CODE / "localize_gpu_camera_remap.py"), "--mode", mode, "--output", str(folder / mode)]
                row = {"trial": trial, "mode": mode, "command": command, "status": "running"}
                state["runs"].append(row)
                atomic_json(args.output / "STATE.json", state)
                with (folder / (mode + "_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output / "STATE.json", state)
                    code = process.wait()
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                atomic_json(args.output / "STATE.json", state)
                if code != 0:
                    raise RuntimeError("A complete original reference localization entry failed")
                results[mode] = read(folder / mode / "RESULT.json")
                guard = read(folder / (mode + "_GUARD.json"))
                whole[mode] = guard["elapsed_seconds"]
                if not all(results[mode]["checks"].values()) or guard["status"] != "completed":
                    raise ValueError("A complete original stage or resource protection failed")
            started = time.perf_counter()
            entries = {mode: Path(results[mode]["localization"]).parent.parent for mode in order}
            protection = compare(entries["native"], entries["CUDA"], entries["native"] / "NEIGHBOR_TRACE.json", entries["CUDA"] / "NEIGHBOR_TRACE.json")
            atomic_json(folder / "INVARIANTS.json", protection)
            comparisons += time.perf_counter() - started
            checks = {"all_original_pixels_features_matches_F_supports_and_poses_protected": protection["all_protected"],
                      "all_sources_and_original_inputs_same": results["native"]["sources"] == results["CUDA"]["sources"] and results["native"]["inputs"] == results["CUDA"]["inputs"],
                      "every_whole_pair_not_over10percent_slower": whole["CUDA"] <= 1.1 * whole["native"]}
            pair = {"trial": trial, "order": order, "checks": checks, "whole_seconds": whole,
                    "complete_reference_preparation_stage_seconds": {mode: result["complete_reference_preparation_stage_seconds"] for mode, result in results.items()}}
            state["pairs"].append(pair)
            atomic_json(args.output / "STATE.json", state)
            print(json.dumps(pair), flush=True)
            if not all(checks.values()):
                raise ValueError("An original complete output or complete pair cost guard failed")
            if trial == 0 and whole["CUDA"] >= whole["native"] and pair["complete_reference_preparation_stage_seconds"]["CUDA"] >= pair["complete_reference_preparation_stage_seconds"]["native"]:
                state.update(status="first_complete_pair_outputs_preserved_both_costs_worse_stop_no_acceptance",
                    complete_output_comparison_seconds=comparisons,
                    whole_before_publication_seconds=time.perf_counter() - STARTED)
                atomic_json(args.output / "RESULT.json", state)
                print(json.dumps({"status": state["status"], "new_optimization_count": 0}), flush=True)
                return
        medians = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("native", "CUDA")}
                   for key in ("whole_seconds", "complete_reference_preparation_stage_seconds")}
        reductions = {key: 1 - row["CUDA"] / row["native"] for key, row in medians.items()}
        state.update(status="three_complete_pairs_preserved_final_cost_pending", raw_medians=medians,
                     raw_reductions=reductions, complete_output_comparison_seconds=comparisons,
                     whole_before_publication_seconds=time.perf_counter() - STARTED)
        atomic_json(args.output / "RESULT.json", state)
        print(json.dumps({"status": state["status"], "raw_reductions": reductions, "new_optimization_count": 0}), flush=True)
    except BaseException as error:
        state.update(status="failed_verification", error=repr(error), whole_before_publication_seconds=time.perf_counter() - STARTED)
        atomic_json(args.output / "FAILURE.json", state)
        raise
    finally:
        atomic_json(args.output / "STATE.json", state)


if __name__ == "__main__":
    main()
