"""Test stable calling costs on every captured original request with fresh workers."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from integer_sift_matching import DistanceWorker
from resident_sift_matching import ResidentWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "requests"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads((args.requests/"REQUESTS.json").read_text(encoding="utf-8"))
    if fixture["status"] != "passed" or len(fixture["records"]) != 48 or not all(fixture["checks"].values()):
        raise ValueError("The complete original actual request capture did not pass")
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in ("descriptor_residency_plan.py",
        "resident_sift_worker.py", "resident_sift_matching.py", "integer_sift_worker.py", "integer_sift_matching.py"))]
    sources = [data_identity(path) for path in paths]
    state = {"status": "running", "sources": sources, "fixture": data_identity(args.requests/"REQUESTS.json"),
        "minimum_original48_actual_call_stage_reduction": .1,
        "no_capacity_formula_format_or_object_parameter_changes": True,
        "complete_native_distance_vectors_indices_and_reverse_exact": True,
        "all_file_read_restore_startup_and_shutdown_reported": True, "pairs": [], "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.perf_counter()
    for trial in range(3):
        folder = args.output/f"trial{trial}"
        folder.mkdir()
        outputs, duration, workers, prep = {}, {}, {}, {}
        order = ("native", "resident") if trial % 2 == 0 else ("resident", "native")
        for mode in order:
            before = time.perf_counter()
            arrays = {}
            for key, item in fixture["blobs"].items():
                path = args.requests/item["file"]
                if data_identity(path) != item["identity"] or path.stat().st_size != item["rows"]*128:
                    raise ValueError("A captured complete descriptor payload changed")
                arrays[key] = np.frombuffer(path.read_bytes(), np.uint8).reshape(item["rows"], 128).astype(np.float32)/512
            prep[mode] = time.perf_counter()-before
            worker = (DistanceWorker if mode == "native" else ResidentWorker)(folder/(mode+"_worker.log"))
            values = []
            try:
                for row in fixture["records"]:
                    values.append(worker.batch(arrays[row["query"]], [arrays[key] for key in row["trains"]]))
            finally:
                worker.close()
            duration[mode] = time.perf_counter()-before
            outputs[mode], workers[mode] = values, worker.receipt
            del arrays
        verify_started = time.perf_counter()
        exact = all(len(left) == len(right) and all(all(np.array_equal(a, b) for a, b in zip(l, r)) for l, r in zip(left, right))
            for left, right in zip(outputs["native"], outputs["resident"]))
        verification = time.perf_counter()-verify_started
        checks = {"all48_complete_real_results_exact": exact and len(outputs["native"]) == len(outputs["resident"]) == 48,
            "both_complete_workers_closed_original_CUDA_budget": all(worker["returncode"] == 0
                and worker["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3 for worker in workers.values()),
            "source_identity_preserved": sources == [data_identity(path) for path in paths]}
        row = {"trial": trial, "order": order, "checks": checks, "complete_component_seconds": duration,
            "data_read_identity_restore_seconds": prep, "pair_full_vector_verification_seconds": verification,
            "calls_seconds": {mode: sum(call["total_prepare_IPC_kernel_seconds"] for call in worker["calls"]) for mode, worker in workers.items()},
            "workers": workers}
        state["pairs"].append(row)
        atomic_json(args.output/"STATE.json", state)
        print({key: value for key, value in row.items() if key != "workers"}, flush=True)
        if not all(checks.values()):
            raise ValueError("An original complete replay output or resource guard failed")
    median = {key: {mode: statistics.median(row[key][mode] for row in state["pairs"]) for mode in ("native", "resident")}
        for key in ("calls_seconds", "complete_component_seconds")}
    reductions = {key: 1-value["resident"]/value["native"] for key, value in median.items()}
    checks = {"all_three_complete_original48_request_pairs_protected": len(state["pairs"]) == 3
            and all(all(row["checks"].values()) for row in state["pairs"]),
        "stable_calls_reduce10percent": reductions["calls_seconds"] >= .1,
        "complete_read_verify_restore_worker_lifecycle_cannot_regress": reductions["complete_component_seconds"] >= 0}
    state.update(status="reliability_passed_original_whole_entry_acceptance_pending" if all(checks.values()) else "failed_verification",
        median=median, reductions=reductions, checks=checks, elapsed_seconds=time.perf_counter()-started)
    atomic_json(args.output/"RESULT.json", state)
    print({key: state[key] for key in ("status", "checks", "median", "reductions")}, flush=True)


if __name__ == "__main__":
    main()
