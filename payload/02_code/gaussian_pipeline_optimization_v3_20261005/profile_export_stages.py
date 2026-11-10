"""Measure unchanged full export stages before choosing a dependency optimization."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import sys

import psutil

from profile_grid_support import original_metric_module
from profile_sparse_ply_oracle import ROOT, V1, V1OUT, native, compare_outputs
from grid_support import grid_support_execution
from segmented_export_execution import segmented_export_execution
from runtime import atomic_json, data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/export_qa_dag_001"
CONTROL = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001/c1/t0/segmented/export"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(BASE)
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    code = Path(__file__).resolve().parent
    module, common = original_metric_module()
    ba = V1OUT / "new_map_metric_BA_001"
    roles = V1OUT / "new_map_roles_postBA_001/TRAINING_ROLES.json"
    images = V1OUT / "e2_geometry_support_001/images"
    sources = [Path(__file__), code / "segmented_export_execution.py", code / "segmented_camera_support.py",
        code / "grid_support.py", code / "native_observation_index.py", code / "binary_map_qa.py",
        code / "profile_grid_support.py", code / "profile_sparse_ply_oracle.py", code / "sparse_ply_oracle.py",
        code / "machine_json.py", code / "run_bounded_journal_task.py", code / "run_with_resource_lease.py",
        Path(native.__file__), V1 / "run_metric_refinement.py", V1 / "runtime.py", Path(module.__file__),
        Path(common.__file__), common.PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json"]
    inputs = [ba / "RESULT.json", roles, *sorted((ba / "weak_position_prior/model").glob("*.bin")),
              *sorted(path for path in images.iterdir() if path.is_file())]
    source_ids = [data_identity(path) for path in sources]
    input_ids = [data_identity(path) for path in inputs]
    copies = output / "source_snapshot"
    copies.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))
    atomic_json(output / "CONTRACT.json", {"sources": source_ids, "inputs": input_ids,
        "unchanged_complete_export_current_R112_R132": True, "no_parallel_implementation_yet": True,
        "saved_output_control_not_a_fresh_performance_pair": str(CONTROL), "new_optimization_count": 0})
    original = {name: getattr(native, name) for name in ("main", "filter_points", "projection_validity", "write_initial_ply", "atomic_json")}
    original_undistort, previous_argv = native.pycolmap.undistort_images, sys.argv
    metric = module.camera_geometry_metrics
    calls = []

    def timed(label, function):
        def wrapper(*values, **options):
            begin = time.perf_counter()
            record = {"stage": label, "ordinal": sum(row["stage"] == label for row in calls), "status": "running"}
            calls.append(record)
            try:
                result = function(*values, **options)
                record["status"] = "completed_original_call"
                if label == "projection_validity":
                    record["result"] = result
                return result
            except BaseException as error:
                record.update(status="failed_original_call", error=repr(error))
                raise
            finally:
                record["seconds"] = time.perf_counter() - begin
        return wrapper

    try:
        for name in ("filter_points", "projection_validity", "write_initial_ply"):
            setattr(native, name, timed(name, original[name]))
        native.pycolmap.undistort_images = timed("undistort_images", original_undistort)
        setup_seconds = time.perf_counter() - STARTED
        with grid_support_execution(module, True) as grid:
            with segmented_export_execution(native, output / "support_states", True) as trace:
                sys.argv = [str(Path(native.__file__)), "--ba", str(ba), "--roles", str(roles),
                            "--images", str(images), "--output", str(output / "export")]
                before = time.perf_counter()
                native.main()
                main_seconds = time.perf_counter() - before
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"status": "failed_original_complete_export_profile", "error": repr(error),
            "calls": calls, "elapsed_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise
    finally:
        for name, function in original.items():
            setattr(native, name, function)
        native.pycolmap.undistort_images, sys.argv = original_undistort, previous_argv
        atomic_json(output / "BINDING_RESTORATION.json", {"original_functions": all(getattr(native, name) is function for name, function in original.items()),
            "original_undistort": native.pycolmap.undistort_images is original_undistort,
            "argv": sys.argv is previous_argv, "metric": module.camera_geometry_metrics is metric, "new_optimization_count": 0})
    closing = time.perf_counter()
    admission = read(output / "export/TRAINING_ADMISSION.json")
    comparison_started = time.perf_counter()
    comparison = compare_outputs(CONTROL, output / "export")
    comparison_seconds = time.perf_counter() - comparison_started
    files = [path for path in (output / "export").rglob("*") if path.is_file()]
    bindings = read(output / "BINDING_RESTORATION.json")
    validities = [row for row in calls if row["stage"] == "projection_validity"]
    checks = {
        "complete_admission_all_original_denominators": admission["status"] == "export_and_admission_verified_training_pending"
            and (admission["full_registered_denominator"], admission["training_count"], admission["training_points"], admission["training_observations"]) == (1021, 1016, 75767, 959120),
        "current_R132_three_support_states3053": [row["stage"] for row in trace["stages"]] == ["before", "after", "final"]
            and [row["images"] for row in trace["stages"]] == [1021, 1016, 1016]
            and all(row["status"] == "complete_original_camera_support_fields" and row["details"]["index"]["complete_native_state_unchanged"] for row in trace["stages"]),
        "all_original_two_filter_two_QA_one_undistort_one_PLY_calls": {label: sum(row["stage"] == label for row in calls)
            for label in ("filter_points", "projection_validity", "undistort_images", "write_initial_ply")}
            == {"filter_points": 2, "projection_validity": 2, "undistort_images": 1, "write_initial_ply": 1}
            and all(row["status"] == "completed_original_call" for row in calls),
        "both_projection_checks_complete_and_valid": len(validities) == 2
            and all(row["result"]["valid"] and row["result"]["observation_count"] == 959120 for row in validities),
        "all_original1050_outputs_and_full_bytes_JSON_control": len(files) == 1050
            and comparison["complete_byte_pairs"] == 1046 and comparison["complete_JSON_pairs"] == 4,
        "sources_and_all_inputs_unchanged": source_ids == [data_identity(path) for path in sources]
            and input_ids == [data_identity(path) for path in inputs],
        "original_real_bindings_and_private_contexts_restored": all(value for key, value in bindings.items() if key != "new_optimization_count")
            and grid["original_binding_restored"] and trace["original_main_restored"],
        "all_private_disk_bounds": sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) <= 30 * 1024**3
            and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
    }
    close_seconds = time.perf_counter() - closing
    result = {"status": "complete_unchanged_export_profile_protected" if all(checks.values()) else "failed_export_profile_protection",
        "checks": checks, "sources": source_ids, "inputs": input_ids, "calls": calls, "support_trace": trace,
        "source_input_setup_seconds": setup_seconds, "original_main_seconds": main_seconds,
        "close_verify_seconds": close_seconds, "full_output_comparison": comparison, "full_output_comparison_seconds": comparison_seconds,
        "complete_entry_before_receipt_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0,
        "profiling_only_no_formal_speed_acceptance": True}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "calls": calls, "original_main_seconds": main_seconds,
                     "complete_entry_before_receipt_seconds": result["complete_entry_before_receipt_seconds"]}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A full original export profile protection failed")


if __name__ == "__main__":
    main()
