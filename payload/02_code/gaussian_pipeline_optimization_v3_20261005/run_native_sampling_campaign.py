"""Measure cold current localization pairs and preserve exact registered outputs."""

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
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001"
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_rows(left, right):
    a, b = (read(p/"localization/H_LOCALIZATION.json") for p in (left, right))
    rows = []
    for x, y in zip(a["rows"], b["rows"]):
        xx = {k: v for k, v in x.items() if k != "image_reference"}
        yy = {k: v for k, v in y.items() if k != "image_reference"}
        rows.append({"name": x["image_name"], "all_non_path_members_exact": xx == yy})
    return {"rows": rows, "all24_registered_members_exact": len(a["rows"]) == len(b["rows"]) == len(rows) == 24 and all(r["all_non_path_members_exact"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.library = args.library.resolve()
    args.output.relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    generic = [read(OWNED/"g1/RESULT.json"), read(OWNED/"g2/RESULT.json")]
    if any(not all(g["checks"].values()) for g in generic):
        raise ValueError("The current complete pixel or actual boundary oracle failed")
    paths = [Path(__file__), CODE/"localize_native_camera_sampling.py", CODE/"native_camera_sampling.py",
             CODE/"camera_pixel_remap.py", CODE/"native_camera_sampling.cpp", args.library,
             CODE/"localize_geometry_sift_fusion.py", CODE/"run_bounded_journal_task.py",
             ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/compare_actual_localization.py"]
    sources = [data_identity(p) for p in paths]
    frozen = {r["path"]: r for g in generic for r in g["sources"]}
    if any(row != frozen[row["path"]] for row in sources if row["path"] in frozen):
        raise ValueError("A current sampling source differs from its complete oracles")
    saved = args.output/"source_snapshot"
    saved.mkdir()
    for ordinal, path in enumerate(paths):
        shutil.copyfile(path, saved/(f"{ordinal:02d}_"+path.name))
    state = {"status": "running", "sources": sources, "runs": [], "pairs": [],
             "minimum_complete_stage_reduction": .05, "maximum_each_whole_ratio": 1.1,
             "whole_median_cannot_regress": True, "new_optimization_count": 0,
             "all_prior_failed_first_build_generic_review_costs_still_pending_six_side_allocation": True,
             "registered_first_pair_stop": "both_phase_and_whole_not_faster_or_any_protection_failure"}
    atomic_json(args.output/"PROTOCOL.json", state)
    comparison_seconds = 0.
    try:
        for trial, order in enumerate((("native", "native_DLL"), ("native_DLL", "native"), ("native", "native_DLL"))):
            folder = args.output/f"t{trial}"
            folder.mkdir()
            results, whole = {}, {}
            for mode in order:
                if time.perf_counter()-STARTED > 1700 or sources != [data_identity(p) for p in paths]:
                    raise ValueError("The bounded campaign or frozen actual source changed")
                side = folder/("n" if mode == "native" else "d")
                command = [sys.executable, "-B", str(CODE/"run_bounded_journal_task.py"),
                    "--receipt", str(folder/(mode+"_GUARD.json")), "--log", str(folder/(mode+".log")),
                    "--maximum-seconds", "450", "--maximum-rss-gib", "8", "--", sys.executable, "-B",
                    str(CODE/"localize_native_camera_sampling.py"), "--mode", mode,
                    "--output", str(side), "--library", str(args.library)]
                row = {"trial": trial, "mode": mode, "command": command, "status": "running"}
                state["runs"].append(row)
                atomic_json(args.output/"STATE.json", state)
                with (folder/(mode+"_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    returncode = process.wait()
                row.update(status="completed" if returncode == 0 else "failed", returncode=returncode)
                atomic_json(args.output/"STATE.json", state)
                if returncode:
                    raise RuntimeError("A complete current localization entry failed")
                results[mode] = read(side/"RESULT.json")
                guard = read(folder/(mode+"_GUARD.json"))
                whole[mode] = guard["elapsed_seconds"]
                if guard["status"] != "completed" or not all(results[mode]["checks"].values()):
                    raise ValueError("A complete current localization or resource guard failed")
            started = time.perf_counter()
            entries = {mode: Path(result["localization"]).parent.parent for mode, result in results.items()}
            protection = compare(entries["native"], entries["native_DLL"], entries["native"]/"NEIGHBOR_TRACE.json", entries["native_DLL"]/"NEIGHBOR_TRACE.json")
            exact = exact_rows(entries["native"], entries["native_DLL"])
            atomic_json(folder/"INVARIANTS.json", {"existing_complete_oracle": protection, "registered_exact_members": exact})
            comparison_seconds += time.perf_counter()-started
            checks = {"complete_pixels_features_matches_K_poses_and_supports": protection["all_protected"],
                      "registered_all_pose_and_stat_members_exact": exact["all24_registered_members_exact"],
                      "all_current_sources_and_inputs_same": results["native"]["sources"] == results["native_DLL"]["sources"] and results["native"]["inputs"] == results["native_DLL"]["inputs"],
                      "each_whole_not_over10percent_slower": whole["native_DLL"] <= whole["native"]*1.1}
            pair = {"trial": trial, "order": order, "checks": checks, "whole_seconds": whole,
                    "complete_reference_preparation_stage_seconds": {mode: r["complete_reference_preparation_stage_seconds"] for mode, r in results.items()}}
            state["pairs"].append(pair)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(pair), flush=True)
            if not all(checks.values()):
                raise ValueError("A complete current entry registered protection failed")
            if trial == 0 and whole["native_DLL"] >= whole["native"] and pair["complete_reference_preparation_stage_seconds"]["native_DLL"] >= pair["complete_reference_preparation_stage_seconds"]["native"]:
                state.update(status="first_complete_pair_preserved_both_costs_not_faster_stop_negative",
                             complete_comparison_seconds=comparison_seconds, whole_before_publication_seconds=time.perf_counter()-STARTED)
                atomic_json(args.output/"RESULT.json", state)
                return
        medians = {key: {mode: statistics.median(pair[key][mode] for pair in state["pairs"]) for mode in ("native", "native_DLL")}
                   for key in ("whole_seconds", "complete_reference_preparation_stage_seconds")}
        state.update(status="three_complete_pairs_preserved_complete_cost_pending", raw_medians=medians,
                     raw_reductions={key: 1-row["native_DLL"]/row["native"] for key, row in medians.items()},
                     complete_comparison_seconds=comparison_seconds, whole_before_publication_seconds=time.perf_counter()-STARTED)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_verification", error=repr(error), whole_before_publication_seconds=time.perf_counter()-STARTED)
        atomic_json(args.output/"FAILURE.json", state)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
