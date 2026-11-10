"""Compare all raw initial outputs and three complete typed trajectory costs."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quality-cache", "oracle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or len(oracle["checks"]) != 73 or not all(oracle["checks"].values()):
        raise ValueError("The complete native typed trajectory oracle failed")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_raw_trajectory.py"), Path(__file__).with_name("typed_odometry_cdr.py"),
        V2/"run_effective_initial.py", V2/"raw_initial_invariants.py", V1/"run_initial_pipeline.py"]
    sources = [data_identity(p) for p in paths]
    if sources[2] != oracle["sources"][1]:
        raise ValueError("The actual typed decoder differs from the full native oracle")
    state = {"status": "running", "sources": sources, "oracle": data_identity(args.oracle/"RESULT.json"), "runs": [], "pairs": [],
        "same_unaccepted_R89_family_not_a_new_variant_count": True, "three_real_pairs": True,
        "minimum_full_pose_and_calibration_stage_reduction": .1, "maximum_complete_entry_ratio": 1.1,
        "first_buffer_allocation_full_covariance_and_native_calibration_in_phase": True,
        "all7338_decisions1002_JPEG11583_pairs_and_full_raw_odometry_exact": True,
        "only_raw_estimate_and_raw_image_quality_in_selection_no_GT_SfM": True,
        "same_current_Q78_Q50_Q53_warm_quality_cache_JPEG_cache_off": True,
        "maximum_group_seconds": 600, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()
    try:
        for trial in range(3):
            folder = args.output/f"trial{trial}"
            folder.mkdir()
            durations, phase = {}, {}
            for mode in (("native", "typed") if trial % 2 == 0 else ("typed", "native")):
                if sources != [data_identity(p) for p in paths] or time.perf_counter()-started > 600:
                    raise ValueError("The actual raw trajectory source or bounded schedule changed")
                target = folder/mode
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(folder/(mode+"_process.json")),
                    "--log", str(folder/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("run_raw_trajectory.py")), "--decoder", mode,
                    "--quality-cache", str(args.quality_cache), "--output", str(target)]
                row = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(row)
                atomic_json(args.output/"STATE.json", state)
                with (folder/(mode+"_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                row.update(returncode=code, status="completed" if code == 0 else "failed")
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("A complete original raw trajectory worker failed")
                receipt = read(target/"initial/SELECTION_RECEIPT.json")
                durations[mode] = read(folder/(mode+"_process.json"))["elapsed_seconds"]
                phase[mode] = receipt["timings"]["raw_pose_and_calibration_seconds"]
            protected = compare_initial_outputs(folder/"native/initial", folder/"typed/initial")
            trace = read(folder/"typed/COMBINATION.json")["execution"]["typed_CDR"]
            records = read(folder/"typed/initial/RAW_SENSOR_CONTRACT.json")["odometry_count"]
            checks = {**protected["checks"], "every_indexed_pose_and_all36_covariance_fields_retained":
                trace["complete_records"] == trace["pose_calls"] == trace["covariance_calls"] == records
                and trace["typed_route_calls"] == 1 and trace["native_route_calls"] == 0
                and trace["owned_typed_buffer_bytes"] == records*352,
                "complete_actual_entry_cost_protected": durations["typed"] <= 1.1*durations["native"]}
            row = {"trial": trial, "complete_entry_seconds": durations, "pose_calibration_seconds": phase,
                "checks": checks, "trace": trace}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(row), flush=True)
            if not all(checks.values()):
                raise ValueError("A full original raw output or whole cost protection failed")
        medians = {key: {mode: statistics.median(p[key][mode] for p in state["pairs"]) for mode in ("native", "typed")}
            for key in ("complete_entry_seconds", "pose_calibration_seconds")}
        reduction = 1-medians["pose_calibration_seconds"]["typed"]/medians["pose_calibration_seconds"]["native"]
        checks = {"three_actual_complete_pairs": len(state["pairs"]) == 3,
            "all_original_outputs_covariance_and_whole_costs_preserved": all(all(p["checks"].values()) for p in state["pairs"]),
            "actual_stage_reduces10percent": reduction >= .1,
            "sources_unchanged": sources == [data_identity(p) for p in paths]}
        state.update(status="complete_raw_typed_stage_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
            checks=checks, median=medians, stage_reduction=reduction,
            real_complete_entry_reduction=1-medians["complete_entry_seconds"]["typed"]/medians["complete_entry_seconds"]["native"],
            elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
