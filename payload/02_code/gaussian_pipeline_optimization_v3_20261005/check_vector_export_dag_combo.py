"""Compare one complete actual DAG pair without treating it as formal performance."""

import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from profile_vector_full_export import ROOT, BASE, read
from profile_sparse_ply_oracle import compare_outputs
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    output = BASE / "d0"
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("profile_vector_export_dag_combo.py")
    rows = {}
    for mode in ("native", "vector"):
        before = time.perf_counter()
        with (output / (mode + ".log")).open("xb") as stream:
            command = [sys.executable, "-B", str(script), "--sampling", mode, "--mode", "dag", "--output", str(output / mode)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            owner = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
            atomic_json(output / (mode + "_OWNER.json"), owner)
            code = process.wait()
        if code: raise RuntimeError("An actual full DAG combo side failed; later sides were not started")
        result = read(output / mode / "RESULT.json")
        if not all(result["checks"].values()): raise AssertionError("An original DAG combo protection failed")
        rows[mode] = {"actual_whole_seconds": time.perf_counter() - before, "result": result, "owner": owner}
    comparison = compare_outputs(output / "native/export", output / "vector/export")
    trace = read(output / "vector/worker/R123_SAMPLING_TRACE.json")
    checks = {"both_actual_DAG_sides_all_original_checks": all(all(r["result"]["checks"].values()) for r in rows.values()),
        "full1050_original_outputs_byte_and_scientific_JSON_exact": comparison["complete_byte_pairs"] == 1046 and comparison["complete_JSON_pairs"] == 4,
        "actual_vector_sampling_one_complete_shared1016_call_and_binding_restore": len(trace["calls"]) == 1 and trace["original_binding_restored"]
            and trace["calls"][0]["trace"]["route"] == "native_full_model_carrier_shared_eight_process_RGB_sampling"
            and all(trace["calls"][0]["trace"]["checks"].values())}
    if not all(checks.values()): raise AssertionError("An actual sampling DAG combo protection failed")
    ratio = rows["vector"]["actual_whole_seconds"] / rows["native"]["actual_whole_seconds"]
    atomic_json(output / "RESULT.json", {"status": "one_complete_R132_R140_sampling_combo_checked_not_formal_performance",
        "checks": checks, "rows": rows, "comparison": comparison, "single_pair_whole_ratio_before_family_fees": ratio,
        "raw_whole_reduction": 1 - ratio, "not_three_pair_medians_or_independent_optimization": True,
        "source": data_identity(Path(__file__)), "complete_seconds": time.perf_counter() - started, "new_optimization_count": 0})
    print(json.dumps({"status": "complete_DAG_combo_checked", "raw_whole_reduction": 1 - ratio, "checks": checks, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
