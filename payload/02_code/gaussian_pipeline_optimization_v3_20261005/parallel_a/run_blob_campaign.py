"""Require three full localization pairs and separately leased verification."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from run_selection_campaign import ROOT, V1, V3, update
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "oracle", "reference"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or not all(oracle["checks"].values()):
        raise ValueError("The complete BLOB lifetime and error oracle did not pass")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("run_blob_case.py"),
        Path(__file__).with_name("feature_blob_views.py"), Path(__file__).with_name("verify_blob_pair.py"),
        V3/"run_with_resource_lease.py", V3/"localize_prepared_queries.py"]
    sources = [data_identity(path) for path in paths]
    if sources[2] not in oracle["sources"]:
        raise ValueError("The actual BLOB implementation differs from its complete oracle")
    state = {"status": "running", "sources": sources, "runs": [], "pairs": [], "counted": False,
        "minimum_complete_read_stage_reduction": .1, "including_pair_verification_reduction_required": .1,
        "maximum_each_whole_entry_ratio": 1.1, "median_whole_cannot_regress": True,
        "unchanged_current_R90_R92_R94_R102_and_all24_48": True,
        "independent_actual_lease_per_run_and_verification": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()

    def run(label, folder, program, options):
        command = [sys.executable, str(V3/"run_with_resource_lease.py"), "--owner", "batch_a_001_PA04_"+label,
            "--receipt", str(folder/(label+"_lease.json")), "--", sys.executable, str(V1/"run_bounded_task.py"),
            "--receipt", str(folder/(label+"_process.json")), "--log", str(folder/(label+".log")),
            "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--", sys.executable, str(program), *options]
        record = {"label": label, "command": command, "status": "running"}
        state["runs"].append(record)
        with (folder/(label+"_controller.log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            atomic_json(args.output/"STATE.json", state)
            update(phase="waiting_or_running_leased_complete_entry", candidate="PA04",
                owned_processes=[{"pid": process.pid, "command": command}], last_receipt=str(folder/(label+"_lease.json")),
                next_step="wait for the exact leased full entry and original output verification")
            code = process.wait()
        record.update(returncode=code, status="completed" if code == 0 else "failed")
        atomic_json(args.output/"STATE.json", state)
        lease = read(folder/(label+"_lease.json"))
        if code or not lease.get("lease_released_utc") or lease.get("returncode") != 0:
            raise RuntimeError("The exact complete entry failed or did not drain its own lease")
        return read(folder/(label+"_process.json"))["elapsed_seconds"]

    try:
        for trial in range(3):
            folder = args.output/str(trial)
            folder.mkdir()
            order = ("copy", "view") if trial % 2 == 0 else ("view", "copy")
            durations, phase, traces = {}, {}, {}
            for mode in order:
                if time.perf_counter()-started > 1800 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The source or bounded schedule changed")
                target = folder/("n" if mode == "copy" else "v")
                durations[mode] = run(str(trial)+"_"+mode, folder, Path(__file__).with_name("run_blob_case.py"),
                    ["--mode", mode, "--output", str(target)])
                traces[mode] = read(target/"BLOB_READ.json")
                phase[mode] = traces[mode]["complete_read_stage_seconds"]
            run(str(trial)+"_verify", folder, Path(__file__).with_name("verify_blob_pair.py"),
                ["--left", str(folder/"n"), "--right", str(folder/"v"),
                    "--reference", str(args.reference), "--output", str(folder/"INVARIANTS.json")])
            verification = read(folder/"INVARIANTS.json")
            checks = {"all_original24_48_matches_poses_features_pixels_inputs_protected": verification["status"] == "passed",
                "all_function_bindings_and_owned_stores_restored": all(row["trace"]["all_function_bindings_restored"]
                    and row["trace"]["instances"] == row["trace"]["all_owned_store_closes"] for row in traces.values()),
                "actual_immutable_views_executed": traces["view"]["trace"]["view_calls"] > 0,
                "whole_entry_pair_cost_protected": durations["view"] <= 1.1*durations["copy"]}
            row = {"trial": trial, "checks": checks, "whole_seconds": durations, "complete_read_seconds": phase,
                "read_plus_independent_pair_verification_seconds": {mode: value+verification["verification_seconds"]/2
                    for mode, value in phase.items()}, "independent_pair_verification_seconds": verification["verification_seconds"],
                "view_trace": traces["view"]["trace"]}
            state["pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(row, flush=True)
            if not all(checks.values()):
                raise ValueError("An actual complete current read stage guard failed")
        median = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("copy", "view")}
            for key in ("whole_seconds", "complete_read_seconds", "read_plus_independent_pair_verification_seconds")}
        reductions = {key: 1-value["view"]/value["copy"] for key, value in median.items()}
        checks = {"all_three_actual_complete_pairs_protected": all(all(row["checks"].values()) for row in state["pairs"]),
            "complete_read_stage_reduction10percent": reductions["complete_read_seconds"] >= .1,
            "including_pair_verification_reduction10percent": reductions["read_plus_independent_pair_verification_seconds"] >= .1,
            "whole_entry_median_cannot_regress": reductions["whole_seconds"] >= 0,
            "sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(status="accepted_pending_monitor" if all(checks.values()) else "rejected_speed", checks=checks,
            median=median, reductions=reductions, elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
        update(phase="PA04_actual_campaign_terminal", candidate="PA04", owned_processes=[], last_receipt=str(args.output/"RESULT.json"),
            next_step="review full cost and external eligibility; continue all unfinished candidates", blocking=None)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        update(phase="PA04_actual_campaign_failed", candidate="PA04", owned_processes=[], last_receipt=str(args.output/"STATE.json"),
            next_step="inspect the exact failure and its drained lease before any further work", blocking=None)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
