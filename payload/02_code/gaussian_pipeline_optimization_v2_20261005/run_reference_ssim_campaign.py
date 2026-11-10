"""Run three isolated actual-evaluator pairs with immutable copied model inputs."""

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
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
    for name in ("runtime", "pair", "localization", "reference-run", "numeric", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    numeric = read(args.numeric/"RESULT.json")
    if not all(numeric["checks"].values()) or numeric["adapter_source"] != data_identity(Path(__file__).with_name("reference_ssim.py")):
        raise ValueError("The current adapter has not passed its official-formula numeric guards")
    if shutil.disk_usage(args.output.parent).free < 20*1024**3:
        raise MemoryError("Insufficient disk headroom")
    localization = read(args.localization)
    intr = localization["PINHOLE_camera"]
    width, height = round(intr["width"]/2), round(intr["height"]/2)
    scales = [width/intr["width"], height/intr["height"], width/intr["width"], height/intr["height"]]
    expected_unique_references = 0
    for row in localization["rows"]:
        window = [round(x*s) for x, s in zip(row["target_window_PINHOLE"], scales)]
        if min(window[2]-window[0], window[3]-window[1]) < 11:
            raise ValueError("All original target windows must remain evaluable")
        expected_unique_references += 1 + int(window != [0, 0, width, height])
    args.output.mkdir(parents=True, exist_ok=False)
    pair = args.output/"pair"
    pair.mkdir()
    paths = [Path(__file__), Path(__file__).with_name("reference_ssim.py"),
             Path(__file__).with_name("evaluate_with_reference_ssim.py"), V1/"evaluate_pair.py"]
    sources = [data_identity(p) for p in paths]
    state = {"status": "preparing_copied_readonly_inputs", "controller_pid": os.getpid(),
        "sources": sources, "input_version": "new_map_admission_v1", "pairs": 3,
        "absolute_SSIM_tolerance": 1e-7, "minimum_SSIM_stage_reduction": .1,
        "all_model_rows_per_run": 48, "all_metric_calls_per_run": 96,
        "expected_unique_reference_views_from_frozen_calibration": expected_unique_references,
        "maximum_process_seconds": 300, "maximum_RSS_bytes": 32*1024**3,
        "maximum_group_disk_bytes": 3*1024**3, "statistics_budget_bytes": 1024**3,
        "inputs": [], "runs": [], "counted": False}
    atomic_json(args.output/"STATE.json", state)
    for label, step in (("reference", 10000), ("combined", 12000)):
        for relative in (Path("STATE.json"), Path("INPUT_MANIFEST.json"),
                Path(f"model/point_cloud/iteration_{step}/point_cloud.ply")):
            src, dst = args.pair/label/relative, pair/label/relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            before = data_identity(src)
            shutil.copyfile(src, dst)
            copied = data_identity(dst)
            if (before["bytes"], before["sha256"]) != (copied["bytes"], copied["sha256"]):
                raise ValueError("Copied evaluator inputs differ from the frozen source")
            state["inputs"].append({"source": before, "copy": copied})
    state["localization"] = data_identity(args.localization)
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()
    for trial in range(3):
        order = ("baseline", "cached") if trial % 2 == 0 else ("cached", "baseline")
        for mode in order:
            if sources != [data_identity(p) for p in paths]:
                raise ValueError("Campaign sources changed during paired measurements")
            folder = f"evaluation_{mode}_{trial}"
            command = [sys.executable, str(V1/"run_bounded_task.py"),
                "--receipt", str(args.output/(folder+"_process.json")),
                "--log", str(args.output/(folder+".log")), "--maximum-seconds", "300",
                "--maximum-rss-gib", "32", "--", sys.executable,
                str(Path(__file__).with_name("evaluate_with_reference_ssim.py")), "--mode", mode,
                "--runtime", str(args.runtime), "--pair", str(pair), "--localization", str(args.localization),
                "--reference-run", str(args.reference_run), "--folder", folder]
            with (args.output/(folder+"_controller.log")).open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                row = {"trial": trial, "mode": mode, "folder": folder, "pid": process.pid, "status": "running", "command": command}
                state["runs"].append(row)
                state.update(status="running", elapsed_seconds=time.monotonic()-started)
                atomic_json(args.output/"STATE.json", state)
                returncode = process.wait()
            row.update(returncode=returncode, status="completed" if returncode == 0 else "failed")
            atomic_json(args.output/"STATE.json", state)
            if returncode:
                raise RuntimeError("An actual evaluation process failed; preserve its output")
            print(json.dumps({"trial": trial, "mode": mode, "status": "completed"}), flush=True)
    checks = {"three_actual_pairs_completed": len(state["runs"]) == 6 and all(r["status"] == "completed" for r in state["runs"]),
              "all_numeric_guards": all(numeric["checks"].values()),
              "sources_unchanged": sources == [data_identity(p) for p in paths]}
    records, times, peaks = [], {"baseline": [], "cached": []}, {"baseline": [], "cached": []}
    for trial in range(3):
        folders = {mode: pair/f"evaluation_{mode}_{trial}" for mode in times}
        evaluations = {mode: read(folder/"EVALUATION.json") for mode, folder in folders.items()}
        profiles = {mode: read(folder/"SSIM_PROFILE.json") for mode, folder in folders.items()}
        row = {"trial": trial, "checks": {}, "maximum_SSIM_delta": 0.}
        for label in ("reference", "combined"):
            a, b = [evaluations[mode]["results"][label]["rows"] for mode in times]
            row["checks"][label+"_all24_kept"] = len(a) == len(b) == 24
            identical, bounded = True, True
            for left, right in zip(a, b):
                identical &= {k: v for k, v in left.items() if k not in ("full_image", "target")} == {k: v for k, v in right.items() if k not in ("full_image", "target")}
                for region in ("full_image", "target"):
                    if region not in left or region not in right:
                        identical = bounded = False
                        continue
                    delta = abs(left[region]["SSIM"]-right[region]["SSIM"])
                    row["maximum_SSIM_delta"] = max(row["maximum_SSIM_delta"], delta)
                    bounded &= delta <= 1e-7
                    identical &= all(left[region][k] == right[region][k] for k in ("PSNR_dB", "MSE"))
            row["checks"][label+"_exact_nonSSIM_results"] = bool(identical)
            row["checks"][label+"_SSIM_within1e_7"] = bool(bounded)
        row["checks"]["six_view_image_bytes_exact"] = data_identity(folders["baseline"]/"SIX_VIEW_COMPARISON.jpg")["sha256"] == data_identity(folders["cached"]/"SIX_VIEW_COMPARISON.jpg")["sha256"]
        for mode, profile in profiles.items():
            times[mode].append(profile["ssim_seconds_including_cold_statistics"])
            peaks[mode].append(profile["peak_CUDA_reserved_bytes"])
            row["checks"][mode+"_all96_calls"] = len(profile["ssim_calls"]) == 96
            row["checks"][mode+"_GPU_budget"] = profile["peak_CUDA_reserved_bytes"] <= profile["CUDA_budget_bytes"]
        cache = profiles["cached"]["cache"]
        row["checks"]["actual_reference_reuse"] = (cache["misses"] == expected_unique_references
            and cache["hits"] == 96-expected_unique_references and cache["evictions"] == 0)
        row["checks"]["cache_bound"] = cache["peak_accounted_bytes"] <= 1024**3
        row["checks"]["paired_quality_guards"] = all(e["development_metric_guard_passed"] for e in evaluations.values())
        records.append(row)
    medians = {mode: statistics.median(values) for mode, values in times.items()}
    gain = 1-medians["cached"]/medians["baseline"]
    checks["all_pair_invariants"] = all(all(r["checks"].values()) for r in records)
    checks["SSIM_stage_improved_at_least10percent"] = gain >= .1
    checks["model_source_and_copy_bytes_preserved"] = all(
        data_identity(Path(item["source"]["path"])) == item["source"] and
        data_identity(Path(item["copy"]["path"])) == item["copy"] for item in state["inputs"])
    accepted = all(checks.values())
    result = {"status": "accepted_stage_pending_round2_combination" if accepted else "failed_stage_acceptance",
        "mechanism": "Q30_REFERENCE_SSIM_REUSE", "checks": checks, "records": records,
        "paired_SSIM_seconds": times, "median_SSIM_seconds": medians, "SSIM_stage_reduction": gain,
        "paired_peak_CUDA_reserved_bytes": peaks, "maximum_SSIM_difference": max(r["maximum_SSIM_delta"] for r in records),
        "elapsed_seconds": time.monotonic()-started, "stage_accepted": accepted,
        "counted_final": False, "round2_combination_accepted": False,
        "training_speed_or_quality_claim": False, "sources": sources}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    state.update(status=result["status"], stage_accepted=accepted, elapsed_seconds=result["elapsed_seconds"])
    atomic_json(args.output/"STATE.json", state)
    print(json.dumps({k: result[k] for k in ("status", "SSIM_stage_reduction", "maximum_SSIM_difference", "checks")}))
    if not accepted:
        raise ValueError("A registered reference-statistic reuse guard failed")


if __name__ == "__main__":
    main()
