"""Run three fresh complete original exports under one measured campaign lease."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import statistics
import subprocess

from profile_sparse_ply_oracle import ROOT, compare_outputs
from runtime import atomic_json, data_identity

CODE = Path(__file__).parent
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001"
PYTHON = ROOT/"05_envs/sfm_py311_v1/Scripts/python.exe"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT)
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "registered_three_fresh_complete_export_pairs", "complete_sides": 0,
             "original_complete_cost_5percent_and_whole_gates": True, "new_optimization_count": 0}
    atomic_json(args.output/"STATE.json", state)
    sources = [Path(__file__), CODE/"profile_segmented_export.py", CODE/"segmented_export_execution.py",
               CODE/"segmented_camera_support.py", CODE/"run_bounded_journal_task.py", CODE/"profile_sparse_ply_oracle.py"]
    source_ids = [data_identity(path) for path in sources]
    pairs = []
    try:
        for ordinal, order in enumerate((("native", "segmented"), ("segmented", "native"), ("native", "segmented"))):
            pair = args.output/f"t{ordinal}"
            pair.mkdir()
            nodes = {}
            for mode in order:
                guard = pair/f"{mode}_guard.json"
                command = [str(PYTHON), "-B", str(CODE/"run_bounded_journal_task.py"), "--receipt", str(guard),
                    "--log", str(pair/f"{mode}.log"), "--maximum-seconds", "1750", "--maximum-rss-gib", "8",
                    "--", str(PYTHON), "-B", str(CODE/"profile_segmented_export.py"),
                    "--mode", mode, "--output", str(pair/mode)]
                atomic_json(pair/f"{mode}_COMMAND.json", {"command": command, "cwd": str(ROOT), "new_optimization_count": 0})
                print(json.dumps({"pair": ordinal, "mode": mode, "status": "starting_complete_export"}), flush=True)
                launched = time.perf_counter()
                process = subprocess.run(command, cwd=ROOT, check=False)
                wrapper_seconds = time.perf_counter()-launched
                resource = read(guard)
                if process.returncode != 0 or resource["returncode"] != 0 or resource["status"] != "completed":
                    raise RuntimeError("A complete original export node failed; later pairs were not started")
                result = read(pair/mode/"RESULT.json")
                if not all(result["checks"].values()):
                    raise AssertionError("A complete original export guard failed")
                # Include startup, final receipt publication and process closure.
                tail = wrapper_seconds-result["whole_before_result_seconds"]
                if tail < 0:
                    raise ValueError("The complete node envelope cannot precede its internal clock")
                nodes[mode] = {"directory": str(pair/mode), "guard": str(guard), "wrapper_seconds": wrapper_seconds,
                    "complete_whole_seconds": wrapper_seconds,
                    "complete_phase_seconds": result["complete_affected_phase_seconds"]+tail,
                    "publication_startup_and_process_close_seconds": tail}
                state.update(status="running_complete_export_pairs", complete_sides=state["complete_sides"]+1,
                             current_pair=ordinal, last_completed_mode=mode)
                atomic_json(args.output/"STATE.json", state)
            begin = time.perf_counter()
            comparison = compare_outputs(pair/"native/export", pair/"segmented/export")
            comparison_seconds = time.perf_counter()-begin
            if comparison["complete_byte_pairs"] != 1046 or comparison["complete_JSON_pairs"] != 4:
                raise AssertionError("The full original export output denominator changed")
            row = {"pair": ordinal, "order": list(order), "nodes": nodes, "comparison": comparison,
                   "comparison_seconds": comparison_seconds,
                   "whole_ratio": nodes["segmented"]["complete_whole_seconds"]/nodes["native"]["complete_whole_seconds"]}
            pairs.append(row)
            atomic_json(pair/"PAIR.json", row)
            if row["whole_ratio"] > 1.1:
                raise AssertionError("The original per-pair complete whole latency guard failed")
            state["completed_pairs"] = len(pairs)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps({"pair": ordinal, "status": "all1050_outputs_exact", "whole_ratio": row["whole_ratio"]}), flush=True)
        if source_ids != [data_identity(path) for path in sources]:
            raise RuntimeError("The frozen campaign sources changed")
        medians = {scope: {mode: statistics.median(pair["nodes"][mode][scope] for pair in pairs)
                          for mode in ("native", "segmented")}
                   for scope in ("complete_phase_seconds", "complete_whole_seconds")}
        result = {"status": "three_complete_export_pairs_exact_final_protections_and_all_cost_pending",
            "pairs": pairs, "sources": source_ids, "medians_before_outer_first_and_review_fees": medians,
            "complete_campaign_before_result_seconds": time.perf_counter()-STARTED,
            "required_error_recovery_external_and_root_review": "pending", "new_optimization_count": 0}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], completed_pairs=3, complete_sides=6)
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({"status": result["status"], "medians": medians}), flush=True)
    except BaseException as error:
        state.update(status="failed_required_complete_campaign_guard", error=repr(error), completed_pairs=len(pairs))
        atomic_json(args.output/"STATE.json", state)
        atomic_json(args.output/"FAILURE.json", {"state": state, "pairs_before_failure": pairs, "sources": source_ids,
            "complete_campaign_before_failure_receipt_seconds": time.perf_counter()-STARTED, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
