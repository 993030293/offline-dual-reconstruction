"""Run one preregistered first pair and resume the remaining two fresh pairs."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from profile_sparse_ply_oracle import ROOT, compare_outputs
from runtime import atomic_json, data_identity

CODE = Path(__file__).resolve().parent
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/export_qa_dag_001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(BASE)
    sources = [Path(__file__), CODE / "profile_export_qa_dag.py", CODE / "export_qa_dag.py",
               CODE / "export_qa_artifact_worker.py", CODE / "run_bounded_journal_task.py", CODE / "run_with_resource_lease.py"]
    source_ids = [data_identity(path) for path in sources]
    contract_path = output / "CONTRACT.json"
    if output.exists():
        if args.through != 3:
            raise FileExistsError("Only the preregistered remaining two pairs may resume an existing campaign")
        contract = read(contract_path)
        if contract["sources"] != source_ids or read(output / "SCREEN.json")["status"] != "complete_first_pair_positive_screen_not_acceptance":
            raise ValueError("A resume requires the exact successful first-pair source and screening contract")
        pairs = [read(output / "t0/PAIR.json")]
        if not pairs[0]["raw_phase_positive"] or pairs[0]["whole_ratio"] >= 1:
            raise ValueError("A nonpositive first pair cannot start further formal trials")
    else:
        output.mkdir(parents=True)
        pairs = []
        atomic_json(contract_path, {"sources": source_ids, "pairs": 3, "orders": [["serial", "dag"], ["dag", "serial"], ["serial", "dag"]],
            "first_pair_is_first_formal_pair_if_all_final_protections_pass": True,
            "complete_affected_phase_speed_minimum": .05, "whole_median_must_not_increase": True,
            "all_necessary_measured_fees_must_be_added_before_acceptance": True, "new_optimization_count": 0})
    prepare_seconds = time.perf_counter() - STARTED
    for ordinal in range(len(pairs), args.through):
        pair = output / f"t{ordinal}"
        pair.mkdir()
        nodes = {}
        order = ("serial", "dag") if ordinal % 2 == 0 else ("dag", "serial")
        for mode in order:
            command = [sys.executable, "-B", str(CODE / "run_bounded_journal_task.py"), "--receipt", str(pair / f"{mode}_GUARD.json"),
                "--log", str(pair / f"{mode}.log"), "--maximum-seconds", "300", "--maximum-rss-gib", "8", "--", sys.executable,
                "-B", str(CODE / "profile_export_qa_dag.py"), "--mode", mode, "--output", str(pair / mode)]
            atomic_json(pair / f"{mode}_COMMAND.json", {"command": command, "cwd": str(ROOT), "new_optimization_count": 0})
            print(json.dumps({"pair": ordinal, "mode": mode, "status": "starting_complete_export"}), flush=True)
            before = time.perf_counter()
            process = subprocess.run(command, cwd=ROOT, check=False)
            seconds = time.perf_counter() - before
            guard = read(pair / f"{mode}_GUARD.json")
            if process.returncode != 0 or guard["status"] != "completed" or guard["returncode"] != 0:
                raise RuntimeError("A full export side failed; remaining sides were not started")
            result = read(pair / mode / "RESULT.json")
            if not all(result["checks"].values()):
                raise AssertionError("A complete original export side protection failed")
            tail = seconds - result["whole_before_result_seconds"]
            if tail < 0:
                raise ValueError("A full process envelope cannot precede its own entry")
            nodes[mode] = {"whole_seconds": seconds, "phase_seconds": result["complete_affected_phase_seconds"] + tail,
                "startup_publication_exit_seconds": tail, "directory": str(pair / mode)}
        before = time.perf_counter()
        comparison = compare_outputs(pair / "serial/export", pair / "dag/export")
        comparison_seconds = time.perf_counter() - before
        if comparison["complete_byte_pairs"] != 1046 or comparison["complete_JSON_pairs"] != 4:
            raise ValueError("The original whole export comparison denominator changed")
        row = {"pair": ordinal, "order": list(order), "nodes": nodes, "comparison": comparison,
            "comparison_seconds": comparison_seconds, "whole_ratio": nodes["dag"]["whole_seconds"] / nodes["serial"]["whole_seconds"],
            "raw_phase_positive": nodes["dag"]["phase_seconds"] < nodes["serial"]["phase_seconds"]}
        atomic_json(pair / "PAIR.json", row)
        pairs.append(row)
        if ordinal == 0 and (not row["raw_phase_positive"] or row["whole_ratio"] >= 1):
            atomic_json(output / "SCREEN.json", {"status": "closed_negative_first_full_pair_no_net_gain", "pairs": pairs,
                "prepare_seconds": prepare_seconds, "complete_entry_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
            return
    result = {"status": "complete_first_pair_positive_screen_not_acceptance" if args.through == 1 else "complete_three_fresh_pairs_cost_review_pending",
        "pairs": pairs, "sources": source_ids, "prepare_seconds": prepare_seconds,
        "this_entry_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
    if args.through == 3:
        control = statistics.median(row["nodes"]["serial"]["phase_seconds"] for row in pairs)
        candidate = statistics.median(row["nodes"]["dag"]["phase_seconds"] for row in pairs)
        result.update(raw_phase_reduction=1 - candidate / control, raw_whole_reduction=1 - statistics.median(row["nodes"]["dag"]["whole_seconds"] for row in pairs)
                      / statistics.median(row["nodes"]["serial"]["whole_seconds"] for row in pairs),
                      formal_supplements_all_measured_fees_and_root_review_required=True)
    atomic_json(output / ("SCREEN.json" if args.through == 1 else "RESULT.json"), result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "pairs")}), flush=True)


if __name__ == "__main__":
    main()
