"""Run a fixed first formal export pair and only its preregistered two successors."""

import time
STARTED = time.perf_counter()

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import psutil

from profile_vector_full_export import ROOT, BASE, read
from profile_sparse_ply_oracle import compare_outputs
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(BASE)
    code = Path(__file__).resolve().parent
    source_paths = [Path(__file__), code / "profile_vector_full_export.py", code / "run_bounded_journal_task.py",
                    code / "run_with_resource_lease.py", code / "profile_sparse_ply_oracle.py", code / "machine_json.py"]
    identities = [data_identity(p) for p in source_paths]
    if output.exists():
        if (args.through != 3 or read(output / "CONTRACT.json")["sources"] != identities
            or read(output / "SCREEN.json")["status"] != "first_full1050_pair_positive_not_acceptance"):
            raise ValueError("Only the exact registered positive first pair may resume")
        for mode in ("native", "vector"):
            saved = read(output / "t0" / mode / "CONTRACT.json")
            for key in ("sources", "inputs"):
                if saved[key] != [data_identity(row["path"]) for row in saved[key]]:
                    raise ValueError("The full first-pair source or input identity changed before resume")
        pairs = [read(output / "t0/PAIR.json")]
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "CONTRACT.json", {"sources": identities, "three_pairs_orders": [["native", "vector"], ["vector", "native"], ["native", "vector"]],
            "first_pair_formal_if_all_final_protections_pass": True, "stage_minimum": .03, "whole_median_ratio_maximum": 1.01,
            "each_whole_ratio_maximum": 1.1, "all_old_and_new_fees_pending": True, "new_optimization_count": 0})
        copies = output / "source_snapshot"
        copies.mkdir()
        for p in source_paths: (copies / p.name).write_bytes(p.read_bytes())
        pairs = []
    prepare_seconds = time.perf_counter() - STARTED
    for ordinal in range(len(pairs), args.through):
        pair = output / ("t" + str(ordinal))
        pair.mkdir()
        rows = {}
        for mode in (("native", "vector") if ordinal % 2 == 0 else ("vector", "native")):
            command = [sys.executable, "-B", str(code / "run_bounded_journal_task.py"), "--receipt", str(pair / (mode + "_GUARD.json")),
                "--log", str(pair / (mode + ".log")), "--maximum-seconds", "300", "--maximum-rss-gib", "8", "--", sys.executable,
                "-B", str(code / "profile_vector_full_export.py"), "--mode", mode, "--output", str(pair / mode)]
            atomic_json(pair / (mode + "_COMMAND.json"), {"command": command, "new_optimization_count": 0})
            before = time.perf_counter()
            process = subprocess.Popen(command, cwd=ROOT)
            identity = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
            atomic_json(pair / (mode + "_OWNER.json"), identity)
            returncode = process.wait()
            elapsed = time.perf_counter() - before
            guard = read(pair / (mode + "_GUARD.json"))
            if returncode or guard["status"] != "completed" or guard["returncode"]:
                raise RuntimeError("A full1050 side failed; remaining sides were not started")
            result = read(pair / mode / "RESULT.json")
            if not all(result["checks"].values()): raise AssertionError("A full1050 side guard failed")
            tail = elapsed - result["whole_before_result_seconds"]
            if tail < 0: raise ValueError("An actual export process envelope precedes its own entry")
            rows[mode] = {"whole_seconds": elapsed, "phase_seconds": result["complete_affected_phase_seconds"] + tail,
                          "startup_result_actual_exit_seconds": tail, "owner": identity}
        before = time.perf_counter()
        comparison = compare_outputs(pair / "native/export", pair / "vector/export")
        compare_seconds = time.perf_counter() - before
        if comparison["complete_byte_pairs"] != 1046 or comparison["complete_JSON_pairs"] != 4:
            raise ValueError("The full1050 comparison denominator changed")
        for row in rows.values():
            row["phase_seconds"] += compare_seconds
            row["whole_seconds"] += compare_seconds
        row = {"pair": ordinal, "rows": rows, "comparison": comparison, "actual_compare_seconds_charged_each_side": compare_seconds,
               "phase_ratio": rows["vector"]["phase_seconds"] / rows["native"]["phase_seconds"],
               "whole_ratio": rows["vector"]["whole_seconds"] / rows["native"]["whole_seconds"]}
        atomic_json(pair / "PAIR.json", row)
        pairs.append(row)
        if ordinal == 0 and (row["phase_ratio"] > .97 or row["whole_ratio"] > 1.1):
            atomic_json(output / "SCREEN.json", {"status": "first_full1050_cost_negative_not_formal_acceptance", "pairs": pairs,
                "prepare_seconds": prepare_seconds, "this_entry_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
            return
    result = {"status": "first_full1050_pair_positive_not_acceptance" if args.through == 1 else "three_full1050_pairs_cost_and_remaining_protections_pending",
              "pairs": pairs, "sources": identities, "prepare_seconds": prepare_seconds,
              "this_entry_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
    if args.through == 3:
        medians = {mode: {key: statistics.median(p["rows"][mode][key] for p in pairs) for key in ("phase_seconds", "whole_seconds")}
                   for mode in ("native", "vector")}
        result.update(medians=medians, raw_stage_reduction=1 - medians["vector"]["phase_seconds"] / medians["native"]["phase_seconds"],
                      raw_whole_reduction=1 - medians["vector"]["whole_seconds"] / medians["native"]["whole_seconds"])
    atomic_json(output / ("SCREEN.json" if args.through == 1 else "RESULT.json"), result)
    print({k: v for k, v in result.items() if k not in ("sources", "pairs")}, flush=True)


if __name__ == "__main__":
    main()
