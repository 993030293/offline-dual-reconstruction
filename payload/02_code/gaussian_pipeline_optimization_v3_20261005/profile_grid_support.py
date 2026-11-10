"""Measure complete original map exports with exact camera grid support."""

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

from grid_support import grid_support_execution
from profile_sparse_ply_oracle import ROOT, V1, V1OUT, compare_outputs, native
from runtime import atomic_json, data_identity


def original_metric_module():
    helpers = ROOT/"02_code/box2_lio_guided_sfm_v1_20261002"
    sys.path.insert(0, str(helpers))
    common = importlib.import_module("common")
    sys.path.insert(0, str(common.PACKAGE/"code/validated_fresh_runtime"))
    module = importlib.import_module("metric_admission")
    expected = common.PACKAGE/"code/validated_fresh_runtime/metric_admission.py"
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError("The actual metric module is not the original frozen package")
    return module, common


def run_leaf(args):
    folder = args.output.resolve()
    folder.relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    identities = contract["sources_and_inputs"]
    paths = [Path(row["path"]) for row in identities]
    module, common = original_metric_module()
    if [data_identity(path) for path in paths] != identities:
        raise RuntimeError("Original frozen source or complete map export input changed")
    prepare = time.perf_counter()-IMPORT_STARTED
    original_writer = native.write_initial_ply
    ba = V1OUT/"new_map_metric_BA_001"
    roles = V1OUT/"new_map_roles_postBA_001/TRAINING_ROLES.json"
    images = V1OUT/"e2_geometry_support_001/images"
    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(native.__file__)), "--ba", str(ba), "--roles", str(roles),
                    "--images", str(images), "--output", str(folder)]
        with grid_support_execution(module, args.leaf_mode == "grid") as trace:
            native.main()
    finally:
        sys.argv = previous_argv
    closing = time.perf_counter()
    receipt = json.loads((folder/"TRAINING_ADMISSION.json").read_text(encoding="utf-8"))
    expected_calls = receipt["full_registered_denominator"]+2*receipt["training_count"]
    if (not trace["original_binding_restored"] or len(trace["calls"]) != expected_calls
            or any(row["status"] != "passed_original_full_camera_support_function" for row in trace["calls"])
            or native.write_initial_ply is not original_writer):
        raise AssertionError("Complete metric call membership or original bindings changed")
    if [data_identity(path) for path in paths] != identities:
        raise RuntimeError("Frozen sources or full input changed during real export")
    if sum(path.stat().st_size for path in folder.parent.rglob("*") if path.is_file()) > 30*1024**3:
        raise MemoryError("Owned complete map exports exceed the output budget")
    if shutil.disk_usage(folder).free < 20*1024**3:
        raise OSError("Original disk headroom violated")
    closing_seconds = time.perf_counter()-closing
    result = {"status": "complete_original_export_and_grid_protections_passed",
              "phase_seconds": sum(row["complete_call_seconds"] for row in trace["calls"])
                  +trace["setup_seconds"]+trace["restore_seconds"]+prepare+closing_seconds,
              "trace": trace, "expected_full_metric_calls": expected_calls,
              "original_sparse_PLY_writer_used": True,
              "complete_leaf_before_receipt_seconds": time.perf_counter()-IMPORT_STARTED,
              "counted": False}
    atomic_json(folder.parent/f"{folder.name}_leaf_RESULT.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leaf-mode", choices=("native", "grid"))
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    if args.leaf_mode is not None:
        if args.contract is None:
            raise ValueError("A real leaf requires the frozen source/input contract")
        run_leaf(args)
        return
    output = args.output.resolve()
    output.relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005")
    output.mkdir(parents=True, exist_ok=False)
    module, common = original_metric_module()
    source_names = ("grid_support.py", "profile_sparse_ply_oracle.py", "sparse_ply_oracle.py", "machine_json.py")
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in source_names),
               Path(native.__file__), V1/"runtime.py", V1/"run_metric_refinement.py",
               ROOT/"02_code/box2_lio_guided_sfm_v1_20261002/common.py", Path(module.__file__),
               common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"]
    ba = V1OUT/"new_map_metric_BA_001"
    inputs = [ba/"RESULT.json", V1OUT/"new_map_roles_postBA_001/TRAINING_ROLES.json",
              *sorted((ba/"weak_position_prior/model").glob("*.bin"))]
    inputs += sorted(path for path in (V1OUT/"e2_geometry_support_001/images").iterdir() if path.is_file())
    identities = [data_identity(path) for path in sources+inputs]
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshot/f"{index:02d}_{path.name}")
    contract = output/"CONTRACT.json"
    atomic_json(contract, {"mechanism": "R112_exact_original_camera_grid_support",
                          "sources_and_inputs": identities, "pairs": 3,
                          "only_original_metric_function_grid_count_replaced": True,
                          "R111_candidate_sparse_writer_enabled": False,
                          "minimum_phase_reduction": .1, "maximum_each_whole_ratio": 1.1,
                          "whole_median_nonregression_required": True,
                          "first_preparation_divisor": 6, "counted": False})
    first_prepare = time.perf_counter()-IMPORT_STARTED
    pairs = []
    try:
        for trial in range(3):
            rows = {}
            order = ("native", "grid") if trial%2 == 0 else ("grid", "native")
            for label in order:
                folder = output/f"pair{trial}_{label.ljust(6, '_')}"
                started = time.perf_counter()
                with (output/f"{folder.name}.log").open("xb") as log:
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--output", str(folder), "--leaf-mode", label, "--contract", str(contract)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                if process.returncode != 0:
                    raise RuntimeError(f"Actual complete original export failed: {label}, {process.returncode}")
                leaf = json.loads((output/f"{folder.name}_leaf_RESULT.json").read_text(encoding="utf-8"))
                if leaf["status"] != "complete_original_export_and_grid_protections_passed":
                    raise AssertionError("Original export or metric source guards did not pass")
                elapsed = time.perf_counter()-started
                external = elapsed-leaf["complete_leaf_before_receipt_seconds"]
                if external < 0:
                    raise AssertionError("Parent envelope omitted part of the real child execution")
                rows[label] = {"whole_seconds": elapsed+first_prepare/6,
                    "phase_seconds": leaf["phase_seconds"]+external+first_prepare/6,
                    "actual_full_metric_calls": leaf["expected_full_metric_calls"],
                    "first_child_start_receipt_and_close_seconds": external,
                    "source_bound_leaf_result": str(output/f"{folder.name}_leaf_RESULT.json"),
                    "folder": str(folder), "child_returncode": process.returncode}
            begin_comparison = time.perf_counter()
            verification = compare_outputs(Path(rows["native"]["folder"]), Path(rows["grid"]["folder"]))
            comparison_fee = time.perf_counter()-begin_comparison
            for row in rows.values():
                row["phase_seconds"] += comparison_fee
                row["whole_seconds"] += comparison_fee
            pair = {"trial": trial, "rows": rows, "verification": verification,
                    "complete_comparison_seconds_charged_each_side": comparison_fee}
            pairs.append(pair)
            atomic_json(output/"STATE.json", {"status": "measuring", "pairs": pairs, "counted": False})
            if rows["grid"]["whole_seconds"] > 1.1*rows["native"]["whole_seconds"]:
                raise RuntimeError("The first required complete whole cost guard failed")
        medians = {label: {field: statistics.median(pair["rows"][label][field] for pair in pairs)
                          for field in ("phase_seconds", "whole_seconds")} for label in ("native", "grid")}
        reduction = 1-medians["grid"]["phase_seconds"]/medians["native"]["phase_seconds"]
        checks = {"three_complete_pairs": len(pairs) == 3,
                  "phase_reduction_at_least10pct": reduction >= .1,
                  "each_whole_at_most1_1": all(pair["rows"]["grid"]["whole_seconds"] <=
                      1.1*pair["rows"]["native"]["whole_seconds"] for pair in pairs),
                  "whole_median_nonregressed": medians["grid"]["whole_seconds"] <= medians["native"]["whole_seconds"],
                  "all_original_export_contents_and_bindings_exact": True}
        result = {"status": "local_complete_grid_cost_guards_passed_pending_external_and_review"
                  if all(checks.values()) else "complete_grid_cost_guard_failed",
                  "checks": checks, "medians": medians, "phase_reduction": reduction,
                  "pairs": pairs, "first_prepare_seconds": first_prepare,
                  "actual_complete_campaign_seconds": time.perf_counter()-IMPORT_STARTED,
                  "counted": False}
        atomic_json(output/"RESULT.json", result)
        print(json.dumps({"status": result["status"], "medians": medians, "phase_reduction": reduction}))
    except BaseException as error:
        atomic_json(output/"FAILURE.json", {"error": repr(error), "pairs": pairs,
                    "actual_complete_campaign_seconds": time.perf_counter()-IMPORT_STARTED,
                    "failed_outputs_preserved": True, "counted": False})
        raise


if __name__ == "__main__":
    main()
