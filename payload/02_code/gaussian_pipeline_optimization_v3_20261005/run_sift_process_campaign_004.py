"""Measure three fresh complete pairs with exact scientific output protection."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001"
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_pair(left, right):
    started = time.perf_counter()
    a, b = read(left/"RESULT.json"), read(right/"RESULT.json")
    entries = [Path(r["localization"]).parent.parent for r in (a, b)]
    protected = compare(*entries, *(p/"NEIGHBOR_TRACE.json" for p in entries))
    rows = [read(r["localization"])["rows"] for r in (a, b)]
    protected["checks"]["all24_complete_numeric_pose_centers_stats_exact"] = [
        {k: v for k, v in row.items() if k != "image_reference"} for row in rows[0]] == [
        {k: v for k, v in row.items() if k != "image_reference"} for row in rows[1]]
    protected["checks"]["all_input_identities_exact"] = a["inputs"] == b["inputs"]
    protected["checks"]["both_full_entry_guards_passed"] = all(all(r["checks"].values()) for r in (a, b))
    protected["all_protected"] = all(protected["checks"].values())
    protected["complete_compare_seconds"] = time.perf_counter()-started
    return protected


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    generic = read(OWNED/"g0/RESULT.json")
    if generic["status"] != "passed" or not all(generic["checks"].values()) or generic["shape_fixture_count"] != 4:
        raise ValueError("The complete original generic contract must pass first")
    paths = [Path(__file__), CODE/"native_sift_process.py", CODE/"localize_sift_process.py",
        CODE/"sift_process_node_003.py", CODE/"sift_process_node_config.py", CODE/"check_native_sift_process_guards.py",
        ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/compare_actual_localization.py",
        CODE/"run_bounded_journal_task.py", CODE/"check_bounded_journal_task_002.py"]
    sources = [data_identity(p) for p in paths]
    if sources[1] != generic["sources"][1] or sources[5] != generic["sources"][0]:
        raise ValueError("A scientific source changed after the generic contract")
    journal = read(OWNED/"j1/RESULT.json")
    if journal["status"] != "passed" or not all(journal["checks"].values()) or sources[7] != journal["sources"][1] or sources[8] != journal["sources"][0]:
        raise ValueError("The immutable resource journal must preserve the original failure and timing contract")
    initial_fee = sum(read(OWNED/name)["child_elapsed_seconds"] for name in ("DIAGNOSTIC_LEASE_001.json", "GENERIC_LEASE_001.json",
        "CAMPAIGN_LEASE_001.json", "CAMPAIGN_LEASE_002.json", "CAMPAIGN_LEASE_003.json",
        "ACTUAL_DIAG_LEASE_001.json", "JOURNAL_LEASE_001.json", "JOURNAL_LEASE_002.json"))
    state = {"status": "running", "sources": sources, "pairs": [], "runs": [], "counted": False,
        "initial_fee_seconds": initial_fee, "orders": [["incumbent", "process"], ["process", "incumbent"], ["incumbent", "process"]],
        "minimum_complete_phase_reduction": .1, "maximum_each_whole_ratio": 1.1,
        "whole_median_must_not_regress": True, "full_scientific_outputs_must_be_exact": True,
        "every_side_private_cold_caches": True, "first_generic_compare_close_fees_split_six_sides": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    try:
        for trial, order in enumerate(state["orders"]):
            folder = args.output/f"t{trial}"
            folder.mkdir()
            costs, nodes = {}, {}
            for mode in order:
                if time.perf_counter()-started > 1700 or sources != [data_identity(p) for p in paths]:
                    raise RuntimeError("A frozen source or campaign time bound changed")
                node, output = folder/("nb" if mode == "incumbent" else "nc"), folder/("b" if mode == "incumbent" else "c")
                command = [sys.executable, "-B", str(CODE/"sift_process_node_003.py"), "--kind", "localization", "--mode", mode,
                    "--node", str(node), "--output", str(output), "--", sys.executable, "-B", str(CODE/"localize_sift_process.py"),
                    "--mode", mode, "--output", str(output)]
                row = {"trial": trial, "mode": mode, "command": command, "status": "running"}
                state["runs"].append(row)
                atomic_json(args.output/"STATE.json", state)
                with (folder/(mode+"_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    atomic_json(args.output/"STATE.json", state)
                    code = process.wait(timeout=1800)
                row.update(status="completed" if code == 0 else "failed", returncode=code)
                atomic_json(args.output/"STATE.json", state)
                nodes[mode] = read(node/"NODE.json")
                if code:
                    raise RuntimeError("A complete actual SIFT process node failed")
                costs[mode] = read(output/"RESULT.json")
            protected = compare_pair(folder/"b", folder/"c")
            atomic_json(folder/"INVARIANTS.json", protected)
            checks = {"all_complete_original_scientific_invariants_exact": protected["all_protected"],
                "all_original_node_guards_closed": all(n["status"] == "completed" and not n["live_Windows_after_wait"]
                    and not any(r["alive"] for r in n["actual_Linux_after_wait"]) for n in nodes.values()),
                "candidate_whole_within1_1": nodes["process"]["complete_node_seconds"] <= 1.1*nodes["incumbent"]["complete_node_seconds"]}
            pair = {"trial": trial, "order": order, "checks": checks,
                "raw_CPU_phase_seconds": {m: c["complete_query_stage_seconds"] for m, c in costs.items()},
                "raw_whole_seconds": {m: n["complete_node_seconds"] for m, n in nodes.items()},
                "complete_compare_seconds": protected["complete_compare_seconds"]}
            state["pairs"].append(pair)
            atomic_json(args.output/"STATE.json", state)
            print(pair, flush=True)
            if not all(checks.values()):
                raise ValueError("An exact output or complete cost guard failed")
        fee = initial_fee+sum(p["complete_compare_seconds"] for p in state["pairs"])
        allocation = fee/6
        for pair in state["pairs"]:
            pair["complete_phase_seconds"] = {m: s+allocation for m, s in pair["raw_CPU_phase_seconds"].items()}
            pair["complete_whole_seconds"] = {m: s+allocation for m, s in pair["raw_whole_seconds"].items()}
        medians = {key: {m: statistics.median(p[key][m] for p in state["pairs"]) for m in ("incumbent", "process")}
            for key in ("complete_phase_seconds", "complete_whole_seconds")}
        reductions = {key: 1-r["process"]/r["incumbent"] for key, r in medians.items()}
        checks = {"all_three_complete_pairs_and_invariants": len(state["pairs"]) == 3 and all(all(p["checks"].values()) for p in state["pairs"]),
            "complete_phase_reduces10percent": reductions["complete_phase_seconds"] >= .1,
            "complete_whole_median_not_regressed": reductions["complete_whole_seconds"] >= 0,
            "frozen_sources_unchanged": sources == [data_identity(p) for p in paths]}
        state.update(status="local_complete_guards_passed_review_pending" if all(checks.values()) else "failed_verification",
            checks=checks, medians=medians, reductions=reductions, complete_first_fees=fee, each_side_fee=allocation,
            complete_campaign_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), complete_campaign_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
