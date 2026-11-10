"""Measure three separately leased complete Oxford predicate components."""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import time

from run_selection_campaign import V1, V3, read, update
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "source-receipt", "oracle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    oracle = read(args.oracle/"RESULT.json")
    if oracle["status"] != "passed" or not all(oracle["checks"].values()):
        raise ValueError("The complete local numerical oracle did not pass")
    paths = [Path(__file__), Path(__file__).with_name("oxford_novelty_case.py"),
        Path(__file__).with_name("spatial_novelty.py"), Path(__file__).with_name("verify_oxford_pair.py"), args.input, args.source_receipt]
    sources = [data_identity(path) for path in paths]
    if sources[2] != oracle["sources"][1]:
        raise ValueError("The public predicate changed after the original numerical oracle")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "sources": sources, "runs": [], "pairs": [], "counted": False,
        "component_only_all2892_raw_body_poses_not_camera_initial_selection": True,
        "fixed_radius_m": .035, "fixed_angle_degrees": 1.2, "minimum_complete_component_reduction": .1,
        "including_verification_reduction_required": .1, "maximum_whole_pair_ratio": 1.1,
        "whole_median_cannot_regress": True, "not_blind_or_GT_evaluated": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()

    def run(label, folder, program, options):
        command = [sys.executable, str(V3/"run_with_resource_lease.py"), "--owner", "batch_a_001_PA02_Oxford_"+label,
            "--receipt", str(folder/(label+"_lease.json")), "--", sys.executable, str(V1/"run_bounded_task.py"),
            "--receipt", str(folder/(label+"_process.json")), "--log", str(folder/(label+".log")),
            "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--", sys.executable, str(program), *options]
        record = {"label": label, "command": command, "status": "running"}
        state["runs"].append(record)
        with (folder/(label+"_controller.log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            atomic_json(args.output/"STATE.json", state)
            update(phase="waiting_or_running_external_component", candidate="PA02", owned_processes=[{"pid": process.pid, "command": command}],
                last_receipt=str(folder/(label+"_lease.json")), next_step="wait for all original Oxford body poses without GT or image selection")
            code = process.wait()
        record.update(status="completed" if code == 0 else "failed", returncode=code)
        atomic_json(args.output/"STATE.json", state)
        lease = read(folder/(label+"_lease.json"))
        if code or lease.get("returncode") != 0 or not lease.get("lease_released_utc"):
            raise ValueError("The complete external component failed or its own lease did not drain")
        return read(folder/(label+"_process.json"))["elapsed_seconds"]

    try:
        for trial in range(3):
            folder = args.output/str(trial)
            folder.mkdir()
            phase, whole = {}, {}
            for mode in (("native", "hash") if trial % 2 == 0 else ("hash", "native")):
                if sources != [data_identity(path) for path in paths] or time.perf_counter()-started > 1800:
                    raise ValueError("An external source or bounded schedule changed")
                whole[mode] = run(str(trial)+"_"+mode, folder, Path(__file__).with_name("oxford_novelty_case.py"),
                    ["--mode", mode, "--input", str(args.input), "--source-receipt", str(args.source_receipt), "--output", str(folder/mode)])
                phase[mode] = read(folder/mode/"RESULT.json")["complete_component_seconds"]
            run(str(trial)+"_verify", folder, Path(__file__).with_name("verify_oxford_pair.py"),
                ["--left", str(folder/"native"), "--right", str(folder/"hash"), "--output", str(folder/"INVARIANTS.json")])
            verification = read(folder/"INVARIANTS.json")
            checks = {"all_original_raw_body_poses_ns_and_predicates_exact": verification["status"] == "passed",
                "complete_whole_pair_cost_protected": whole["hash"] <= 1.1*whole["native"]}
            state["pairs"].append({"trial": trial, "checks": checks, "whole_seconds": whole, "complete_component_seconds": phase,
                "component_plus_verification_seconds": {mode: value+verification["verification_seconds"]/2 for mode, value in phase.items()},
                "verification_seconds": verification["verification_seconds"]})
            atomic_json(args.output/"STATE.json", state)
            if not all(checks.values()):
                raise ValueError("An actual external component output or whole cost guard failed")
        median = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("native", "hash")}
            for key in ("whole_seconds", "complete_component_seconds", "component_plus_verification_seconds")}
        reductions = {key: 1-value["hash"]/value["native"] for key, value in median.items()}
        checks = {"all_three_actual_component_pairs_protected": all(all(row["checks"].values()) for row in state["pairs"]),
            "complete_component_reduction10percent": reductions["complete_component_seconds"] >= .1,
            "including_all_independent_verification_reduction10percent": reductions["component_plus_verification_seconds"] >= .1,
            "whole_median_cannot_regress": reductions["whole_seconds"] >= 0,
            "sources_unchanged": sources == [data_identity(path) for path in paths]}
        state.update(status="external_component_passed" if all(checks.values()) else "external_component_rejected_speed", checks=checks,
            median=median, reductions=reductions, elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        update(phase="PA02_external_component_terminal", candidate="PA02", owned_processes=[], last_receipt=str(args.output/"STATE.json"),
            next_step="review exact external scope and finish batch recovery and source decisions", blocking=None)
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
