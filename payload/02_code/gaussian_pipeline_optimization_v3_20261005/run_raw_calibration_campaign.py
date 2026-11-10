"""Compare complete raw selection and three actual calibration reuse triplets."""

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
    if oracle["status"] != "passed" or len(oracle["checks"]) != 12 or not all(oracle["checks"].values()):
        raise ValueError("The complete native calibration oracle failed")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_raw_calibration.py"),
        Path(__file__).with_name("calibration_body_cache.py"), V2/"run_effective_initial.py",
        V2/"raw_initial_invariants.py", V1/"run_initial_pipeline.py"]
    sources = [data_identity(path) for path in paths]
    if sources[2] != oracle["sources"][1]:
        raise ValueError("The actual calibration decoder differs from the full native oracle")
    state = {"status": "running", "sources": sources, "oracle": data_identity(args.oracle/"RESULT.json"),
        "runs": [], "triplets": [], "minimum_complete_calibration_stage_reduction": .1,
        "maximum_complete_entry_ratio": 1.1, "minimum_stage_relative_to_untouched_incumbent": True,
        "first_schema_and_source_composition_setup_added_to_original_stage": True,
        "all_original_frames_JPEG_pairs_and_full_raw_odometry_exact": True,
        "only_raw_estimate_and_raw_image_quality_no_GT_or_SfM": True,
        "same_current_Q78_Q50_Q53_warm_quality_cache_JPEG_cache_off": True,
        "maximum_group_seconds": 600, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()
    try:
        orders = (("incumbent", "native", "memo"), ("memo", "incumbent", "native"),
            ("native", "memo", "incumbent"))
        for trial, order in enumerate(orders):
            folder = args.output/f"trial{trial}"
            folder.mkdir()
            durations, phases, stage_detail, traces = {}, {}, {}, {}
            for mode in order:
                if sources != [data_identity(path) for path in paths] or time.perf_counter()-started > 600:
                    raise ValueError("The raw calibration source or bounded schedule changed")
                target = folder/mode
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(folder/(mode+"_process.json")),
                    "--log", str(folder/(mode+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("run_raw_calibration.py")), "--calibration-mode", mode,
                    "--quality-cache", str(args.quality_cache), "--output", str(target)]
                run = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(run)
                atomic_json(args.output/"STATE.json", state)
                with (folder/(mode+"_controller.log")).open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    run["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait()
                run.update(returncode=code, status="completed" if code == 0 else "failed")
                atomic_json(args.output/"STATE.json", state)
                if code:
                    raise RuntimeError("A complete raw calibration worker failed")
                receipt = read(target/"initial/SELECTION_RECEIPT.json")
                trace = read(target/"COMBINATION.json")["execution"].get("calibration_body")
                composed = read(target/"CALIBRATION.json")["additional_source_composition_seconds"]
                detail = {"original_phase": receipt["timings"]["raw_pose_and_calibration_seconds"],
                    "decoder_schema_setup": trace["setup_seconds"] if trace else 0.,
                    "additional_source_composition": composed}
                phases[mode] = sum(detail.values())
                stage_detail[mode] = detail
                traces[mode] = trace
                durations[mode] = read(folder/(mode+"_process.json"))["elapsed_seconds"]
            protected = {mode: compare_initial_outputs(folder/"incumbent/initial", folder/mode/"initial")
                for mode in ("native", "memo")}
            record_count = oracle["actual_count"]
            native, memo = traces["native"], traces["memo"]
            checks = {"native_all_original_outputs_preserved": protected["native"]["all_protected"],
                "memo_all_original_outputs_preserved": protected["memo"]["all_protected"],
                "all_original_native_decodes_retained": native["calls"] == native["native_calls"] == record_count
                    and native["hits"] == 0 and traces["incumbent"] is None,
                "each_memo_record_has_fresh_header_or_native_full_decode": memo["calls"] == record_count
                    and memo["hits"]+memo["native_calls"] == record_count and memo["native_calls"] >= 1,
                "bounded_private_calibration_cache": memo["maximum_entries"] <= 16
                    and memo["cache_key_bytes"] <= 16*(65536+4),
                "actual_candidate_complete_entry_cost_protected": durations["memo"] <= 1.1*durations["incumbent"]}
            row = {"trial": trial, "order": order, "complete_entry_seconds": durations,
                "complete_pose_calibration_seconds": phases, "stage_detail": stage_detail,
                "traces": traces, "checks": checks, "output_protections": protected}
            state["triplets"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(row), flush=True)
            if not all(checks.values()):
                raise ValueError("A complete output or original entry cost protection failed")
        medians = {key: {mode: statistics.median(row[key][mode] for row in state["triplets"])
            for mode in ("incumbent", "native", "memo")}
            for key in ("complete_entry_seconds", "complete_pose_calibration_seconds")}
        reduction = 1-medians["complete_pose_calibration_seconds"]["memo"]/medians["complete_pose_calibration_seconds"]["incumbent"]
        checks = {"three_actual_complete_triplets": len(state["triplets"]) == 3,
            "all_outputs_and_real_complete_cost_protected": all(all(row["checks"].values()) for row in state["triplets"]),
            "actual_complete_stage_reduces10percent_against_incumbent": reduction >= .1,
            "sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(status="complete_raw_calibration_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
            checks=checks, median=medians, stage_reduction=reduction,
            real_complete_entry_reduction=1-medians["complete_entry_seconds"]["memo"]/medians["complete_entry_seconds"]["incumbent"],
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
