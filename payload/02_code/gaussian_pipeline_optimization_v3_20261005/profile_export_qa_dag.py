"""Execute the complete current map export with serial or owned DAG scheduling."""

import time
STARTED = time.perf_counter()

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys

import psutil

from profile_grid_support import original_metric_module
from profile_sparse_ply_oracle import ROOT, V1, V1OUT, native
from grid_support import grid_support_execution
from segmented_export_execution import segmented_export_execution
from export_qa_dag import export_qa_dag_execution
from runtime import atomic_json, data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/export_qa_dag_001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serial", "dag"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(BASE)
    output.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    code = Path(__file__).resolve().parent
    module, common = original_metric_module()
    ba = V1OUT / "new_map_metric_BA_001"
    roles = V1OUT / "new_map_roles_postBA_001/TRAINING_ROLES.json"
    images = V1OUT / "e2_geometry_support_001/images"
    sources = [Path(__file__), code / "export_qa_dag.py", code / "export_qa_artifact_worker.py",
        code / "segmented_export_execution.py", code / "segmented_camera_support.py", code / "grid_support.py",
        code / "native_observation_index.py", code / "binary_map_qa.py", code / "profile_grid_support.py",
        code / "profile_sparse_ply_oracle.py", code / "sparse_ply_oracle.py", code / "machine_json.py",
        Path(native.__file__), V1 / "run_metric_refinement.py", V1 / "runtime.py", Path(module.__file__),
        Path(common.__file__), common.PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json"]
    inputs = [ba / "RESULT.json", roles, *sorted((ba / "weak_position_prior/model").glob("*.bin")),
              *sorted(path for path in images.iterdir() if path.is_file())]
    source_ids, input_ids = [data_identity(path) for path in sources], [data_identity(path) for path in inputs]
    copies = output / "source_snapshot"
    copies.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))
    atomic_json(output / "CONTRACT.json", {"mode": args.mode, "sources": source_ids, "inputs": input_ids,
        "all_original_quality_and1050_outputs": True, "common_accepted_R112_R132": True,
        "whole_affected_region_including_undistortion_and_artifacts": True, "new_optimization_count": 0})
    original = {name: getattr(native, name) for name in ("main", "filter_points", "projection_validity", "write_initial_ply", "atomic_json")}
    original_source = inspect.getsource(native.main)
    metric, undistort, previous = module.camera_geometry_metrics, native.pycolmap.undistort_images, sys.argv
    setup_seconds = time.perf_counter() - STARTED
    trace = None
    try:
        with grid_support_execution(module, True) as grid:
            with segmented_export_execution(native, output / "support_states", True) as support:
                with export_qa_dag_execution(native, original_source, args.mode, output / "worker") as trace:
                    sys.argv = [str(Path(native.__file__)), "--ba", str(ba), "--roles", str(roles),
                                "--images", str(images), "--output", str(output / "export")]
                    before = time.perf_counter()
                    native.main()
                    main_seconds = time.perf_counter() - before
        sys.argv = previous
        before_close = time.perf_counter()
        admission = read(output / "export/TRAINING_ADMISSION.json")
        files = [path for path in (output / "export").rglob("*") if path.is_file()]
        worker = trace["worker"]
        worker_checks = worker is None
        if args.mode == "dag":
            result = worker["worker_result"]
            worker_checks = worker["status"] == "completed_and_joined" and worker["closed"] and worker["returncode"] == worker["closed_returncode"] == 0
            worker_checks = worker_checks and (result["registered"], result["points"], result["observations"]) == (1016, 75767, 959120)
            try:
                process_gone = psutil.Process(worker["identity"]["pid"]).create_time() != worker["identity"]["creation_time"]
            except psutil.NoSuchProcess:
                process_gone = True
            worker_checks = worker_checks and process_gone
        checks = {
            "original_admission_all1021_1016_75767_959120": admission["status"] == "export_and_admission_verified_training_pending"
                and (admission["full_registered_denominator"], admission["training_count"], admission["training_points"], admission["training_observations"]) == (1021, 1016, 75767, 959120),
            "complete3053_support_three_unchanged_states": [row["stage"] for row in support["stages"]] == ["before", "after", "final"]
                and [row["images"] for row in support["stages"]] == [1021, 1016, 1016]
                and all(row["status"] == "complete_original_camera_support_fields" and row["details"]["index"]["complete_native_state_unchanged"] for row in support["stages"]),
            "full_affected_region_one_start_one_end": trace["region_start_count"] == trace["region_end_count"] == 1,
            "owned_worker_original_calls_protected_and_actual_exit": worker_checks,
            "all_original1050_outputs_four_JSON": len(files) == 1050 and sum(path.suffix == ".json" for path in files) == 4,
            "all_original_source_input_identities_unchanged": source_ids == [data_identity(path) for path in sources]
                and input_ids == [data_identity(path) for path in inputs],
            "all_original_bindings_and_contexts_restored": all(getattr(native, name) is function for name, function in original.items())
                and native.pycolmap.undistort_images is undistort and module.camera_geometry_metrics is metric and sys.argv is previous
                and grid["original_binding_restored"] and support["original_main_restored"] and trace["original_binding_restored"],
            "all_owned_disk_bounds": sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) <= 30 * 1024**3
                and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
        }
        close_seconds = time.perf_counter() - before_close
        phase = setup_seconds + trace["affected_region_seconds"] + close_seconds
        phase += sum(row[key] for row in (grid, support, trace) for key in ("setup_seconds", "restore_seconds"))
        result = {"status": "complete_original_export_DAG_guarded" if all(checks.values()) else "failed_complete_export_DAG_guard",
            "mode": args.mode, "checks": checks, "sources": source_ids, "inputs": input_ids, "admission": admission,
            "trace": trace, "support_trace": support, "source_input_setup_seconds": setup_seconds,
            "close_verify_seconds": close_seconds, "complete_affected_phase_seconds": phase,
            "original_main_seconds": main_seconds, "whole_before_result_seconds": time.perf_counter() - STARTED,
            "new_optimization_count": 0}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks, "phase_seconds": phase,
                          "whole_before_result_seconds": result["whole_before_result_seconds"]}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A required original full export protection failed")
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"status": "failed_complete_export_DAG_no_acceptance", "error": repr(error),
            "trace": trace, "elapsed_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise
    finally:
        sys.argv = previous
        atomic_json(output / "BINDING_RESTORATION.json", {"original_functions": all(getattr(native, name) is function for name, function in original.items()),
            "original_undistort": native.pycolmap.undistort_images is undistort, "argv": sys.argv is previous,
            "metric": module.camera_geometry_metrics is metric, "new_optimization_count": 0})


if __name__ == "__main__":
    main()
