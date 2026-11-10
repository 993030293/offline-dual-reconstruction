"""Compare complete current-map exports with unchanged serial and DAG operations."""

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from atomic_json_windows import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"))
from profile_sparse_ply_oracle import compare_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    original = read_json(args.side / "export_wrapper.json")
    arguments = original["arguments"]
    image_root = Path(arguments[arguments.index("--images") + 1])
    ba_root = Path(arguments[arguments.index("--ba") + 1])
    roles = Path(arguments[arguments.index("--roles") + 1])
    admission = read_json(args.side / "export/TRAINING_ADMISSION.json")
    atomic_json(args.output / "CONTRACT.json", {"complete_current_image_root": str(image_root),
        "current_BA_root": str(ba_root), "roles": str(roles), "pairs": 3,
        "common_R112_R132_both_sides": True, "quality_and_geometric_parameters_unchanged": True,
        "primary_export_untouched": True, "raw_native_output_membership_and_all_values_checked": True,
        "other_CPU_mapping_can_overlap_timings_are_engineering_diagnostics": True})
    started, rows = time.monotonic(), []
    try:
        for trial in range(3):
            pair = args.output / ("pair" + str(trial))
            pair.mkdir()
            entries = {}
            for mode in (("serial", "dag") if trial % 2 == 0 else ("dag", "serial")):
                folder = pair / mode
                folder.mkdir()
                command = [sys.executable, str(CODE / "run_owned_task.py"), "--receipt", str(folder / "PROCESS.json"),
                    "--log", str(folder / "LOG.txt"), "--maximum-seconds", "600", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(CODE / "run_native_stage.py"), "--kind", "export", "--mode", "optimized",
                    "--raw", str(args.raw), "--export-qa-mode", mode, "--ba", str(ba_root), "--roles", str(roles),
                    "--images", str(image_root), "--output", str(folder / "export")]
                if read_json(ba_root / "RESULT.json")["selected"] == "pure_sim3_alignment":
                    command.append("--allow-pure-sim3-recovery")
                before = time.monotonic()
                run = subprocess.run(command, capture_output=True, text=True,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
                measured = read_json(folder / "PROCESS.json")
                if run.returncode or measured["status"] != "completed":
                    raise RuntimeError("Full current export DAG stage failed: " + str(folder))
                result = read_json(folder / "export/TRAINING_ADMISSION.json")
                if (result["training_count"], result["training_points"], result["training_observations"]) != (
                    admission["training_count"], admission["training_points"], admission["training_observations"]):
                    raise AssertionError("The complete admission denominator changed")
                wrapper = read_json(folder / "export_wrapper.json")
                trace = wrapper["module_trace"]["R140"]
                if not trace["original_binding_restored"] or not (trace["region_start_count"] == trace["region_end_count"] == 1):
                    raise AssertionError("A DAG export boundary did not close")
                if mode == "dag" and not (trace["worker"]["closed"] and trace["worker"]["closed_returncode"] == 0):
                    raise AssertionError("The native artifact worker did not drain")
                entries[mode] = {"whole_process_seconds": time.monotonic() - before,
                    "affected_region_seconds": trace["affected_region_seconds"], "complete_trace": trace,
                    "output": str(folder / "export")}
            comparison = compare_outputs(Path(entries["serial"]["output"]), Path(entries["dag"]["output"]))
            rows.append({"trial": trial, "entries": entries, "full_output_comparison": comparison})
            atomic_json(args.output / "STATE.json", {"status": "running", "rows": rows})
        controls = statistics.median(row["entries"]["serial"]["whole_process_seconds"] for row in rows)
        candidates = statistics.median(row["entries"]["dag"]["whole_process_seconds"] for row in rows)
        atomic_json(args.output / "RESULT.json", {"status": "complete_current_export_DAG_outcomes_recorded",
            "rows": rows, "median_whole_process_reduction": 1 - candidates / controls,
            "all_export_files_and_scientific_values_equal": True, "elapsed_seconds": time.monotonic() - started,
            "isolated_speed_or_four_case_generalization_claim": False})
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"status": "failed_complete_export_DAG_check", "error": repr(error),
                                                 "rows": rows, "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
