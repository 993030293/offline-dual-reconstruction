"""Measure complete fresh native camera-path exports and independent output checks."""

import argparse
import inspect
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
import export_new_camera_path as native
from runtime import atomic_json, data_identity
from machine_json import machine_json_execution, exact_json_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "04_outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    dataset = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
    raw = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p1_raw_combined_001/frame_decisions.jsonl"
    source_paths = [Path(__file__), Path(__file__).with_name("machine_json.py"),
                    Path(native.__file__), V1 / "runtime.py"]
    input_paths = [raw, *(dataset / "sparse/0" / name for name in
                            ("cameras.bin", "images.bin", "points3D.bin"))]
    prep = time.perf_counter()
    identities = [data_identity(p) for p in source_paths + input_paths]
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for p in source_paths:
        shutil.copyfile(p, snapshot / p.name)
    atomic_json(args.output / "CONTRACT.json", {"status": "frozen_before_measurement",
        "sources_and_inputs": identities, "complete_native_entry": str(Path(native.__file__)),
        "phase": "all_native_JSON_writes_full_parse_exact_validation_setup_restore_pair_check",
        "original_all1016_projection_and_time_and_input_guards": True,
        "stage_minimum_reduction": .1, "each_whole_maximum_ratio": 1.1,
        "median_whole_nonregression_required": True, "new_training_or_math": False,
        "counted": False})
    first_prep_seconds = time.perf_counter() - prep
    original_argv = sys.argv
    pairs = []
    try:
        for trial in range(3):
            pair = args.output / ("trial" + str(trial))
            rows = {}
            for mode in (("native", "compact") if trial % 2 == 0 else ("compact", "native")):
                output = pair / mode
                whole = time.perf_counter()
                before = [data_identity(p) for p in source_paths + input_paths]
                if before != identities:
                    raise ValueError("A complete native export input or source changed")
                with machine_json_execution(native, mode == "compact") as trace:
                    sys.argv = [str(Path(native.__file__)), "--dataset", str(dataset),
                                "--raw-decisions", str(raw), "--output", str(output)]
                    native.main()
                if len(trace["calls"]) != 2 or not trace["original_binding_restored"]:
                    raise AssertionError("A complete native output or binding was lost")
                if [data_identity(p) for p in source_paths + input_paths] != identities:
                    raise ValueError("An input changed during complete native export")
                phase = trace["setup_seconds"] + trace["restore_seconds"] + sum(
                    r["write_seconds"] + r["complete_validation_seconds"] for r in trace["calls"])
                rows[mode] = {"whole_seconds": time.perf_counter() - whole,
                    "phase_seconds": phase, "trace": trace}
                print(json.dumps({"trial": trial, "mode": mode, **rows[mode]}), flush=True)
            compare = time.perf_counter()
            for name in ("camera_path.json", "RECEIPT.json"):
                first = json.loads((pair / "native" / name).read_text(encoding="utf-8"))
                second = json.loads((pair / "compact" / name).read_text(encoding="utf-8"))
                exact_json_values(first, second)
            compare_seconds = time.perf_counter() - compare
            for row in rows.values():
                row["phase_seconds"] += compare_seconds + first_prep_seconds / 6
                row["whole_seconds"] += compare_seconds + first_prep_seconds / 6
            pairs.append({"trial": trial, "rows": rows, "paired_check_seconds_charged_each_side": compare_seconds,
                "all1016_paths_and_complete_RECEIPT_exact": True})
            atomic_json(args.output / "STATE.json", {"status": "measuring", "pairs": pairs,
                "first_preparation_seconds": first_prep_seconds, "counted": False})
        medians = {mode: {kind: statistics.median(p["rows"][mode][kind] for p in pairs)
            for kind in ("phase_seconds", "whole_seconds")} for mode in ("native", "compact")}
        checks = {"phase_reduction_at_least10pct":
            medians["compact"]["phase_seconds"] <= .9 * medians["native"]["phase_seconds"],
            "each_complete_whole_at_most1_1": all(p["rows"]["compact"]["whole_seconds"] <=
                 1.1 * p["rows"]["native"]["whole_seconds"] for p in pairs),
            "complete_whole_median_nonregressed": medians["compact"]["whole_seconds"] <=
                 medians["native"]["whole_seconds"],
            "all_source_and_input_identities_current": identities ==
                 [data_identity(p) for p in source_paths + input_paths]}
        result = {"status": "native_export_cost_passed_browser_and_generic_pending" if all(checks.values()) else
            "complete_native_export_cost_gate_failed", "pairs": pairs, "medians": medians,
            "phase_reduction": 1 - medians["compact"]["phase_seconds"] / medians["native"]["phase_seconds"],
            "whole_reduction": 1 - medians["compact"]["whole_seconds"] / medians["native"]["whole_seconds"],
            "checks": checks, "first_preparation_seconds": first_prep_seconds,
            "complete_elapsed_seconds": time.perf_counter() - started,
            "py_json_encoder_current_source": inspect.getsource(json.JSONEncoder.iterencode), "counted": False}
        atomic_json(args.output / "RESULT.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("pairs", "py_json_encoder_current_source")}), flush=True)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
