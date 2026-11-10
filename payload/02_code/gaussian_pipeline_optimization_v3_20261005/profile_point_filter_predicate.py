"""Measure R113 with full original exports and actual process-close accounting."""

import time

IMPORT_STARTED = time.perf_counter()

import argparse
import importlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np
import pycolmap

from point_filter_predicate import ROOT, V1, native, point_filter_execution
from point_filter_oracles import compare_outputs
from runtime import atomic_json, data_identity

V1OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005"
OWNED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/point_filter_predicate_001"


def original_metric_module():
    sys.path.insert(0, str(ROOT / "02_code/box2_lio_guided_sfm_v1_20261002"))
    common = importlib.import_module("common")
    sys.path.insert(0, str(common.PACKAGE / "code/validated_fresh_runtime"))
    module = importlib.import_module("metric_admission")
    if Path(module.__file__).resolve() != (common.PACKAGE / "code/validated_fresh_runtime/metric_admission.py").resolve():
        raise ValueError("The original frozen metric module was not loaded")
    return module, common


def input_paths():
    ba = V1OUT / "new_map_metric_BA_001"
    return [ba / "RESULT.json", V1OUT / "new_map_roles_postBA_001/TRAINING_ROLES.json",
            *sorted((ba / "weak_position_prior/model").glob("*.bin")),
            *sorted(path for path in (V1OUT / "e2_geometry_support_001/images").rglob("*") if path.is_file())]


def check_contract(contract):
    if [str(path.resolve()) for path in input_paths()] != contract["complete_input_membership"]:
        raise RuntimeError("Complete original input membership changed")
    identities = contract["sources_and_inputs"]
    if [data_identity(Path(row["path"])) for row in identities] != identities:
        raise RuntimeError("Frozen source or complete export input changed")
    if pycolmap.__version__ != contract["pycolmap"] or np.__version__ != contract["numpy"]:
        raise RuntimeError("The frozen local runtime version changed")


def run_leaf(args):
    folder = args.output.resolve()
    folder.relative_to(OWNED)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    module, _ = original_metric_module()
    check_contract(contract)
    prepare = time.perf_counter() - IMPORT_STARTED
    original_writer, original_metric = native.write_initial_ply, module.camera_geometry_metrics
    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(native.__file__)), "--ba", str(V1OUT / "new_map_metric_BA_001"),
                    "--roles", str(V1OUT / "new_map_roles_postBA_001/TRAINING_ROLES.json"),
                    "--images", str(V1OUT / "e2_geometry_support_001/images"), "--output", str(folder)]
        with point_filter_execution(native, args.leaf_mode == "columns") as trace:
            native.main()
    finally:
        sys.argv = previous_argv
    closing = time.perf_counter()
    if (not trace["original_binding_restored"] or len(trace["calls"]) != 2
            or any(row["status"] != "original_filter_completed" for row in trace["calls"])
            or native.write_initial_ply is not original_writer or module.camera_geometry_metrics is not original_metric):
        raise AssertionError("Original full filter membership or bindings changed")
    if args.leaf_mode == "columns" and any(row["route"] != "columns" for row in trace["calls"]):
        raise AssertionError("A supported actual original-main invocation silently abstained")
    check_contract(contract)
    if sum(path.stat().st_size for path in OWNED.rglob("*") if path.is_file()) > 30 * 1024**3:
        raise MemoryError("Owned complete exports exceed the original output budget")
    if shutil.disk_usage(folder).free < 20 * 1024**3:
        raise OSError("Original disk headroom violated")
    closing_seconds = time.perf_counter() - closing
    result = {"status": "complete_original_export_and_filter_protections_passed", "trace": trace,
              "phase_seconds": sum(row["complete_call_seconds"] for row in trace["calls"])
                  + trace["setup_seconds"] + trace["restore_seconds"] + prepare + closing_seconds,
              "prepare_seconds": prepare, "closing_checks_seconds": closing_seconds,
              "original_sparse_PLY_writer_and_grid_metric_used": True,
              "complete_leaf_before_receipt_seconds": time.perf_counter() - IMPORT_STARTED,
              "counted": False}
    atomic_json(folder.parent / f"{folder.name}_leaf_RESULT.json", result)


def summarize(pairs):
    medians = {label: {field: statistics.median(pair["rows"][label][field] for pair in pairs)
                      for field in ("phase_seconds", "whole_seconds")} for label in ("native", "columns")}
    reduction = 1 - medians["columns"]["phase_seconds"] / medians["native"]["phase_seconds"]
    checks = {"three_complete_pairs": len(pairs) == 3, "phase_reduction_at_least10pct": reduction >= .1,
              "each_whole_at_most1_1": all(pair["rows"]["columns"]["whole_seconds"] <=
                  1.1 * pair["rows"]["native"]["whole_seconds"] for pair in pairs),
              "whole_median_nonregressed": medians["columns"]["whole_seconds"] <= medians["native"]["whole_seconds"],
              "all_original_export_contents_and_bindings_exact": True}
    return medians, reduction, checks


def close_actual_process(args):
    output = args.output.resolve()
    output.relative_to(OWNED)
    lease = json.loads(args.close_lease.read_text(encoding="utf-8"))
    result = json.loads((output / "PAIRS_BEFORE_PROCESS_CLOSE.json").read_text(encoding="utf-8"))
    if lease.get("returncode") != 0 or not lease.get("lease_released_utc") or lease.get("status") != "child_exited":
        raise RuntimeError("The actual measured campaign has not closed and released its lease successfully")
    actual_components = result["first_prepare_seconds"] + sum(pair["actual_comparison_seconds"] +
        sum(row["actual_child_envelope_seconds"] for row in pair["rows"].values()) for pair in result["pairs"])
    unallocated = lease["child_elapsed_seconds"] - actual_components
    if unallocated < 0:
        raise AssertionError("The recorded whole components exceed the actual closed campaign")
    for pair in result["pairs"]:
        for row in pair["rows"].values():
            row["phase_seconds"] += unallocated / 6
            row["whole_seconds"] += unallocated / 6
            row["actual_campaign_bookkeeping_publication_and_close_seconds_charged"] = unallocated / 6
    medians, reduction, checks = summarize(result["pairs"])
    result.update(status="local_complete_filter_cost_guards_passed_pending_external_and_review"
                  if all(checks.values()) else "complete_filter_cost_guard_failed", checks=checks,
                  medians=medians, phase_reduction=reduction, actual_process_close_accounted=True,
                  actual_campaign_seconds_from_released_lease=lease["child_elapsed_seconds"],
                  complete_unallocated_publication_and_close_seconds=unallocated,
                  lease_receipt=str(args.close_lease.resolve()), external_component_run=False, counted=False)
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "medians": medians, "phase_reduction": reduction}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--leaf-mode", choices=("native", "columns"))
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--close-lease", type=Path)
    args = parser.parse_args()
    if args.close_lease:
        close_actual_process(args)
        return
    if args.leaf_mode:
        if args.contract is None:
            raise ValueError("Every real leaf requires the complete source/input contract")
        run_leaf(args)
        return
    if args.oracle is None:
        raise ValueError("The actual native generic oracle must precede measurement")
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    if oracle.get("status") != "generic_native_point_filter_oracle_passed":
        raise RuntimeError("The required native generic oracle has not passed")
    output = args.output.resolve()
    output.relative_to(OWNED)
    output.mkdir(parents=True, exist_ok=False)
    module, common = original_metric_module()
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in
               ("point_filter_predicate.py", "point_filter_oracles.py", "check_point_filter_predicate.py")),
               Path(native.__file__), V1 / "runtime.py", V1 / "run_metric_refinement.py",
               ROOT / "02_code/box2_lio_guided_sfm_v1_20261002/common.py", Path(module.__file__),
               common.PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json",
               Path(pycolmap.__file__), Path(pycolmap._core.__file__), Path(np._core._multiarray_umath.__file__)]
    oracle_contract = json.loads(args.oracle.with_name("CONTRACT.json").read_text(encoding="utf-8"))
    if [data_identity(Path(row["path"])) for row in oracle_contract["sources_and_inputs"]] != oracle_contract["sources_and_inputs"]:
        raise RuntimeError("The source-bound oracle or its complete fixtures changed")
    inputs = input_paths()
    identities = [data_identity(path) for path in sources + inputs]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(sources):
        if path.suffix == ".py" or path.suffix == ".json":
            shutil.copyfile(path, snapshot / f"{index:02d}_{path.name}")
    contract = output / "CONTRACT.json"
    atomic_json(contract, {"mechanism": "R113_exact_live_point_filter_predicate", "sources_and_inputs": identities,
                          "complete_input_membership": [str(path.resolve()) for path in inputs], "pairs": 3,
                          "pycolmap": pycolmap.__version__, "numpy": np.__version__, "oracle": str(args.oracle.resolve()),
                          "only_original_filter_points_predicates_replaced": True,
                          "R111_and_R112_candidate_functions_enabled": False, "minimum_phase_reduction": .1,
                          "maximum_each_whole_ratio": 1.1, "whole_median_nonregression_required": True,
                          "first_preparation_divisor": 6, "counted": False})
    first_prepare = time.perf_counter() - IMPORT_STARTED
    pairs = []
    try:
        for trial in range(3):
            rows = {}
            order = ("native", "columns") if trial % 2 == 0 else ("columns", "native")
            for label in order:
                folder = output / f"pair{trial}_{label}"
                atomic_json(output / "STATE.json", {"status": "actual_leaf_running", "trial": trial, "label": label,
                                                      "completed_pairs": pairs, "counted": False})
                started = time.perf_counter()
                with (output / f"{folder.name}.log").open("xb") as log:
                    process = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(folder),
                        "--leaf-mode", label, "--contract", str(contract)], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                if process.returncode != 0:
                    raise RuntimeError(f"Actual original full export failed: {label}, {process.returncode}")
                leaf_path = output / f"{folder.name}_leaf_RESULT.json"
                leaf = json.loads(leaf_path.read_text(encoding="utf-8"))
                if leaf["status"] != "complete_original_export_and_filter_protections_passed":
                    raise AssertionError("The actual full filter/export protections did not pass")
                elapsed = time.perf_counter() - started
                external = elapsed - leaf["complete_leaf_before_receipt_seconds"]
                if external < 0:
                    raise AssertionError("The envelope omitted actual leaf publication or process close")
                rows[label] = {"whole_seconds": elapsed + first_prepare / 6,
                    "phase_seconds": leaf["phase_seconds"] + external + first_prepare / 6,
                    "actual_child_envelope_seconds": elapsed, "actual_complete_filter_calls": len(leaf["trace"]["calls"]),
                    "startup_wait_receipt_and_child_close_seconds": external, "source_bound_leaf_result": str(leaf_path),
                    "folder": str(folder), "child_returncode": process.returncode}
            beginning = time.perf_counter()
            verification = compare_outputs(Path(rows["native"]["folder"]), Path(rows["columns"]["folder"]))
            fee = time.perf_counter() - beginning
            for row in rows.values():
                row["phase_seconds"] += fee
                row["whole_seconds"] += fee
            pairs.append({"trial": trial, "rows": rows, "verification": verification,
                          "actual_comparison_seconds": fee, "complete_comparison_seconds_charged_each_side": fee})
            atomic_json(output / "STATE.json", {"status": "complete_pair_checked", "pairs": pairs, "counted": False})
            if rows["columns"]["whole_seconds"] > 1.1 * rows["native"]["whole_seconds"]:
                raise RuntimeError("The first required complete whole cost guard failed")
        atomic_json(output / "PAIRS_BEFORE_PROCESS_CLOSE.json", {
            "status": "complete_pairs_pending_actual_process_close", "pairs": pairs,
            "first_prepare_seconds": first_prepare, "before_final_publication_seconds": time.perf_counter() - IMPORT_STARTED,
            "counted": False})
        print(json.dumps({"status": "complete_pairs_pending_actual_process_close", "pairs": len(pairs)}))
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"error_type": type(error).__name__, "error": str(error), "pairs": pairs,
                    "actual_complete_campaign_seconds": time.perf_counter() - IMPORT_STARTED,
                    "negative_preserved": True, "counted": False})
        raise


if __name__ == "__main__":
    main()
