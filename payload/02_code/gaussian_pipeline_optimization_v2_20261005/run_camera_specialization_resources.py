"""Compare three complete Scene trials for independently scoped RGB specializations."""

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
from run_prefetch_resources import pair_checks


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("specialized_training_camera.py"),
        Path(__file__).with_name("load_with_camera_specialization.py"),
        Path(__file__).with_name("validate_camera_prefetch.py"), Path(__file__).with_name("training_camera_metadata.py")]
    sources = [data_identity(p) for p in paths]
    import shutil
    (args.output/"sources").mkdir()
    for path in paths:
        shutil.copyfile(path, args.output/"sources"/path.name)
    state = {"status": "running", "sources": sources, "runs": [], "pairs": [], "counted": False,
        "three_actual_trials_per_mechanism": True, "all1016_effective_pixels_K_views_Gaussians_and_RNG_exact": True,
        "alpha_oracle": "original all-ones mask and absent mask have the same opaque RGB semantics; actual all-ones tensor checked",
        "minimum_stage_gain": .1, "complete_Scene_ratio_max": 1.1,
        "opaque_gate": "peak Scene RSS reduction", "clampfree_gate": "actual native clamp allocation phase reduction",
        "projection_gate": "legacy and calibrated projection phase reduction including first matrix construction",
        "CPU_images_unchanged": True, "same_common_Q31_prefetch": True,
        "Q52_metadata_view_disabled_on_both_sides_after_failed_training_quality": True,
        "maximum_group_seconds": 1200, "maximum_RSS_GiB": 32, "full_training_quality_and_combination_required": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()
    try:
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            order = ["baseline", "opaque", "clampfree", "projection"] if trial % 2 == 0 else ["projection", "clampfree", "opaque", "baseline"]
            for mode in order:
                if sources != [data_identity(p) for p in paths] or time.monotonic()-started > 1200:
                    raise ValueError("A camera resource implementation or bounded schedule changed")
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "300", "--maximum-rss-gib", "32", "--",
                    sys.executable, str(Path(__file__).with_name("load_with_camera_specialization.py")),
                    "--camera-mode", mode, "--runtime", str(args.runtime), "--dataset", str(args.dataset), "--output", str(pair/mode)]
                with (pair/(mode+"_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    row = {"trial": trial, "mode": mode, "pid": process.pid, "status": "running"}
                    state["runs"].append(row)
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(returncode=code, status="completed" if code == 0 else "failed")
                if code:
                    raise RuntimeError("A full native camera resource process failed")
            baseline = read(pair/"baseline/RESULT.json")
            for mode in ("opaque", "clampfree", "projection"):
                candidate = read(pair/mode/"RESULT.json")
                checks = pair_checks(baseline, candidate)
                if mode == "projection":
                    checks.pop("same_GPU_payload")
                    checks["shared_projection_values_exact_and_CUDA_allocation_not_increased"] = checks["rows_exact"] and candidate["peak_CUDA_reserved_bytes"] <= baseline["peak_CUDA_reserved_bytes"]
                traces = {label: value["camera_specialization"][0]["specialization"] for label, value in (("baseline", baseline), (mode, candidate))}
                checks["all_native_camera_calls_and_RGB_contract"] = all(t["native_calls"] == 1016 for t in traces.values())
                item = {"trial": trial, "mode": mode, "checks": checks,
                    "all_protected": all(checks.values()), "Scene_seconds": {"baseline": baseline["Scene_load_seconds"], "candidate": candidate["Scene_load_seconds"]},
                    "peak_RSS_bytes": {"baseline": baseline["peak_RSS_after_load_bytes"], "candidate": candidate["peak_RSS_after_load_bytes"]},
                    "phase_seconds": {"baseline": traces["baseline"]["alpha" if mode == "opaque" else "clamp"] if mode != "projection" else traces["baseline"]["legacy_projection"]+traces["baseline"]["calibrated_projection"],
                        "candidate": traces[mode]["alpha" if mode == "opaque" else "clamp"] if mode != "projection" else traces[mode]["legacy_projection"]+traces[mode]["calibrated_projection"]},
                    "traces": traces, "complete_time_protected": candidate["Scene_load_seconds"] <= 1.1*baseline["Scene_load_seconds"]}
                state["pairs"].append(item)
                atomic_json(args.output/"STATE.json", state)
                print(json.dumps({key: item[key] for key in ("trial", "mode", "all_protected", "Scene_seconds", "phase_seconds", "peak_RSS_bytes", "complete_time_protected")}), flush=True)
                if not item["all_protected"] or not item["complete_time_protected"]:
                    raise ValueError("A complete actual RGB camera guard failed")
        results = {}
        for mode in ("opaque", "clampfree", "projection"):
            pairs = [p for p in state["pairs"] if p["mode"] == mode]
            median = {key: {label: statistics.median(p[key][label] for p in pairs) for label in ("baseline", "candidate")}
                for key in ("Scene_seconds", "peak_RSS_bytes", "phase_seconds")}
            metric = "peak_RSS_bytes" if mode == "opaque" else "phase_seconds"
            reduction = 1-median[metric]["candidate"]/median[metric]["baseline"]
            results[mode] = {"status": "resource_passed_training_quality_pending" if reduction >= .1 else "rejected_insufficient_effect",
                "median": median, "stage_reduction": reduction, "absolute_phase_seconds_saved": median["phase_seconds"]["baseline"]-median["phase_seconds"]["candidate"],
                "pairs": pairs, "counted": False}
        atomic_json(args.output/"RESULT.json", {"status": "resource_campaign_completed_training_quality_pending", "results": results,
            "sources": sources, "sources_unchanged": sources == [data_identity(p) for p in paths], "counted": False})
        state.update(status="resource_campaign_completed_training_quality_pending", elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
