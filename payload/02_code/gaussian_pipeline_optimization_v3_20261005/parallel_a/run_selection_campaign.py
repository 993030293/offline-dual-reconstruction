"""Require three current full raw pairs with individual exclusive lease receipts."""

import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V3 = ROOT/"02_code/gaussian_pipeline_optimization_v3_20261005"
DOC = ROOT/"research/2026-10-05-gaussian-optimization-round3-50/parallel_001/a"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update(**fields):
    path = DOC/"STATUS.json"
    payload = read(path)
    payload.update(fields, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(path, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "oracle", "quality-cache", "reference-initial"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or not all(oracle["checks"].values()):
        raise ValueError("The complete generic selector oracle did not pass")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_selection_case.py"), Path(__file__).with_name("spatial_novelty.py"),
        Path(__file__).with_name("verify_initial_pair.py"), V1/"selection_engine.py", V1/"quality_engine.py", V3/"run_with_resource_lease.py"]
    sources = [data_identity(path) for path in paths]
    if sources[2] != oracle["sources"][1] or sources[4] != oracle["sources"][2]:
        raise ValueError("The current selector differs from its complete generic oracle")
    state = {"status": "running", "sources": sources, "runs": [], "pairs": [], "counted": False,
        "three_actual_pairs": True, "minimum_complete_selector_reduction": .1,
        "maximum_each_whole_entry_ratio": 1.1, "whole_entry_median_cannot_regress": True,
        "all_index_setup_original_validation_return_and_close_in_phase": True,
        "independent_full_output_verification_cost_recorded_separately_and_also_disclosed_with_phase": True,
        "all_original7338_6797_1002_11583_and48_exclusions": True, "cache_miss_rejects_writes_to_frozen_cache": True,
        "each_run_and_pair_verification_uses_separate_actual_resource_lease": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()

    def run(label, folder, program, options):
        command = [sys.executable, str(V3/"run_with_resource_lease.py"), "--owner", "batch_a_001_PA02_"+label,
            "--receipt", str(folder/(label+"_lease.json")), "--", sys.executable, str(V1/"run_bounded_task.py"),
            "--receipt", str(folder/(label+"_process.json")), "--log", str(folder/(label+".log")),
            "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--", sys.executable, str(program), *options]
        record = {"label": label, "command": command, "status": "running"}
        state["runs"].append(record)
        with (folder/(label+"_controller.log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            atomic_json(args.output/"STATE.json", state)
            update(phase="waiting_or_running_leased_complete_entry", candidate="PA02",
                owned_processes=[{"pid": process.pid, "command": command}], last_receipt=str(folder/(label+"_lease.json")),
                next_step="wait for this exact leased run; do not restart on observation timeout")
            code = process.wait()
        record.update(returncode=code, status="completed" if code == 0 else "failed")
        atomic_json(args.output/"STATE.json", state)
        if code:
            raise RuntimeError("A complete selector entry or resource lease failed")
        lease = read(folder/(label+"_lease.json"))
        if not lease.get("lease_released_utc") or lease.get("returncode") != 0:
            raise ValueError("The exact measured entry did not finish and release its lease")
        return read(folder/(label+"_process.json"))["elapsed_seconds"]

    try:
        for trial in range(3):
            folder = args.output/str(trial)
            folder.mkdir()
            order = ("native", "hash") if trial % 2 == 0 else ("hash", "native")
            durations, phase, traces, receipts = {}, {}, {}, {}
            for mode in order:
                if time.perf_counter()-started > 1800 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The source or complete bounded schedule changed")
                durations[mode] = run(str(trial)+"_"+mode, folder, Path(__file__).with_name("run_selection_case.py"),
                    ["--mode", mode, "--quality-cache", str(args.quality_cache), "--output", str(folder/mode)])
                traces[mode] = read(folder/mode/"SELECTOR.json")
                receipts[mode] = read(folder/mode/"initial/SELECTION_RECEIPT.json")
                phase[mode] = (receipts[mode]["timings"]["selection_seconds"]+traces[mode]["additional_complete_selector_lifecycle_seconds"]
                    +traces[mode]["complete_original_frame_decisions_manifests_and_final_state_output_seconds"]
                    +read(folder/mode/"COLLECTION_COST.json")["selector_trace_output_seconds"])
            run(str(trial)+"_verify", folder, Path(__file__).with_name("verify_initial_pair.py"),
                ["--left", str(folder/"native/initial"), "--right", str(folder/"hash/initial"),
                    "--reference", str(args.reference_initial), "--output", str(folder/"INVARIANTS.json")])
            verification = read(folder/"INVARIANTS.json")
            checks = {"all_current_original_raw_decisions_K_poses_images_pairs_exact": verification["status"] == "passed",
                "all7338_cache_hits_and_no_old_cache_writes_both": all(row["timings"]["raw_image_scoring_detail"]["cache_hits"] == 7338 for row in receipts.values()),
                "all_owned_source_and_function_bindings_restored": all(row["trace"]["aliases_restored"] and row["timing_bindings_restored"] for row in traces.values()),
                "actual_hash_novelty_queries_executed": traces["hash"]["trace"]["hash_queries"] > 0,
                "whole_entry_pair_cost_protected": durations["hash"] <= 1.1*durations["native"]}
            row = {"trial": trial, "checks": checks, "whole_seconds": durations, "complete_selector_seconds": phase,
                "complete_selector_plus_independent_pair_verification_seconds": {mode: value+verification["verification_seconds"]/2 for mode, value in phase.items()},
                "independent_pair_verification_seconds": verification["verification_seconds"], "hash_trace": traces["hash"]["trace"]}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(row, flush=True)
            if not all(checks.values()):
                raise ValueError("An actual complete current selector guard failed")
        median = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("native", "hash")}
            for key in ("whole_seconds", "complete_selector_seconds", "complete_selector_plus_independent_pair_verification_seconds")}
        reductions = {key: 1-value["hash"]/value["native"] for key, value in median.items()}
        checks = {"all_three_complete_current_pairs_preserved": all(all(row["checks"].values()) for row in state["pairs"]),
            "complete_selector_stage_reduction10percent": reductions["complete_selector_seconds"] >= .1,
            "including_all_disclosed_pair_verification_reduction10percent": reductions["complete_selector_plus_independent_pair_verification_seconds"] >= .1,
            "whole_entry_median_cannot_regress": reductions["whole_seconds"] >= 0,
            "sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(status="accepted_pending_monitor" if all(checks.values()) else "rejected_speed", checks=checks,
            median=median, reductions=reductions, elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
        update(phase="PA02_actual_campaign_terminal", candidate="PA02", owned_processes=[], last_receipt=str(args.output/"RESULT.json"),
            next_step="review the complete current result; continue PA01 PA03 PA04", blocking=None)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        update(phase="PA02_actual_campaign_failed", candidate="PA02", owned_processes=[], last_receipt=str(args.output/"STATE.json"),
            next_step="inspect exact failure evidence and release before further work", blocking=None)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
