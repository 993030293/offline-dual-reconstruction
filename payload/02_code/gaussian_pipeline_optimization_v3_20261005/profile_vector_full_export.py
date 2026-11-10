"""Execute unchanged whole-map export with optional strict shared SIMD sampling."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import sys

import psutil

from profile_grid_support import original_metric_module
from profile_sparse_ply_oracle import ROOT, V1, V1OUT, native
from grid_support import grid_support_execution
from segmented_export_execution import segmented_export_execution
from dataset_sampling_shared_carrier import cached_dataset
from vector_camera_sampling import library_supports_simd
from runtime import atomic_json, data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001/v5_simd_001"
LIBRARY = BASE / "b0/native_vector_camera_sampling.dll"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "vector"), required=True)
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
    sources = [Path(__file__), *[code / name for name in (
        "dataset_sampling_shared_carrier.py", "dataset_sampling_carrier.py", "complete_carrier_configuration.py",
        "vector_camera_sampling.py", "native_vector_camera_sampling.cpp", "native_camera_sampling.py", "native_camera_sampling.cpp",
        "camera_pixel_remap.py", "segmented_export_execution.py", "segmented_camera_support.py", "grid_support.py",
        "native_observation_index.py", "binary_map_qa.py", "profile_grid_support.py", "profile_sparse_ply_oracle.py",
        "sparse_ply_oracle.py", "machine_json.py")], LIBRARY, Path(native.__file__), V1 / "run_metric_refinement.py",
        V1 / "runtime.py", Path(module.__file__), Path(common.__file__), common.PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json"]
    inputs = [ba / "RESULT.json", roles, *sorted((ba / "weak_position_prior/model").glob("*.bin")),
              *sorted(path for path in images.iterdir() if path.is_file())]
    source_ids, input_ids = [data_identity(p) for p in sources], [data_identity(p) for p in inputs]
    copies = output / "source_snapshot"
    copies.mkdir()
    for index, path in enumerate(sources):
        if path.suffix in (".py", ".cpp", ".json"):
            shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))
    atomic_json(output / "CONTRACT.json", {"mode": args.mode, "sources": source_ids, "inputs": input_ids,
        "common_R112_R132_original_all1050": True, "R113_optional_disabled_R140_combination_pending": True,
        "original_undistort_argument_contract": {"output_type": "COLMAP", "num_patch_match_src_images": 0, "jpeg_quality": 97, "num_threads": 8},
        "new_optimization_count": 0})
    originals = {name: getattr(native, name) for name in ("main", "filter_points", "projection_validity", "write_initial_ply", "atomic_json")}
    metric, undistort, previous = module.camera_geometry_metrics, native.pycolmap.undistort_images, sys.argv
    calls = []

    def sampling(destination, model, input_images, **kwargs):
        begin = time.perf_counter()
        if set(kwargs) != {"image_names", "output_type", "num_patch_match_src_images", "jpeg_quality", "num_threads"}:
            raise ValueError("The original complete undistortion signature changed")
        if any(kwargs[k] != v for k, v in {"output_type": "COLMAP", "num_patch_match_src_images": 0, "jpeg_quality": 97, "num_threads": 8}.items()):
            raise ValueError("The original complete undistortion options changed")
        record = {"status": "running", "original_names": list(kwargs["image_names"]), "mode": args.mode}
        calls.append(record)
        try:
            if args.mode == "vector" and library_supports_simd(LIBRARY):
                # The carrier invokes the unchanged original seed writer internally.
                native.pycolmap.undistort_images = undistort
                try:
                    record["trace"] = cached_dataset(destination, model, input_images, kwargs["image_names"], LIBRARY, output / "sampling_proof")
                finally:
                    native.pycolmap.undistort_images = sampling
            else:
                undistort(destination, model, input_images, **kwargs)
                record["trace"] = {"route": "original_complete_native_undistortion", "closed": True}
            record["status"] = "complete_original_work"
        except BaseException as error:
            record.update(status="failed_original_boundary", error=repr(error))
            raise
        finally:
            record["seconds"] = time.perf_counter() - begin

    native.pycolmap.undistort_images = sampling
    setup_seconds = time.perf_counter() - STARTED
    try:
        with grid_support_execution(module, True) as grid:
            with segmented_export_execution(native, output / "support_states", True) as support:
                sys.argv = [str(Path(native.__file__)), "--ba", str(ba), "--roles", str(roles),
                            "--images", str(images), "--output", str(output / "export")]
                before = time.perf_counter()
                native.main()
                main_seconds = time.perf_counter() - before
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"status": "failed_full1050_source_or_sampling_boundary", "calls": calls,
            "error": repr(error), "complete_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise
    finally:
        restore_start = time.perf_counter()
        sys.argv, native.pycolmap.undistort_images = previous, undistort
        restore_seconds = time.perf_counter() - restore_start
        atomic_json(output / "BINDING_RESTORATION.json", {"original_functions": all(getattr(native, name) is function for name, function in originals.items()),
            "original_undistort": native.pycolmap.undistort_images is undistort, "argv": sys.argv is previous,
            "metric": module.camera_geometry_metrics is metric, "new_optimization_count": 0})
    close_start = time.perf_counter()
    admission = read(output / "export/TRAINING_ADMISSION.json")
    files = [p for p in (output / "export").rglob("*") if p.is_file()]
    bindings = read(output / "BINDING_RESTORATION.json")
    vector_trace = calls[0]["trace"] if len(calls) == 1 else {}
    checks = {
        "original_admission_all1021_1016_75767_959120": admission["status"] == "export_and_admission_verified_training_pending"
            and (admission["full_registered_denominator"], admission["training_count"], admission["training_points"], admission["training_observations"]) == (1021, 1016, 75767, 959120),
        "complete3053_support_states_and_original_native_state": [r["stage"] for r in support["stages"]] == ["before", "after", "final"]
            and [r["images"] for r in support["stages"]] == [1021, 1016, 1016]
            and all(r["status"] == "complete_original_camera_support_fields" and r["details"]["index"]["complete_native_state_unchanged"] for r in support["stages"]),
        "actual_one_complete1016_sampling_call_and_all_original_options": len(calls) == 1 and len(calls[0]["original_names"]) == 1016
            and calls[0]["status"] == "complete_original_work" and vector_trace["closed"],
        "actual_vector_one_map_eight_workers1015_native_seed1_closed": args.mode == "native" or
            (vector_trace["route"] == "native_full_model_carrier_shared_eight_process_RGB_sampling"
             and len(vector_trace["workers"]) == 8 and sum(r["count"] for r in vector_trace["workers"]) == 1015
             and sum(r["sampler_trace"]["sampler_calls"] for r in vector_trace["workers"]) == 1015
             and all(vector_trace["checks"].values())),
        "all_original1050_outputs_four_JSON": len(files) == 1050 and sum(p.suffix == ".json" for p in files) == 4,
        "all_sources_and1028_inputs_unchanged": len(input_ids) == 1028 and source_ids == [data_identity(p) for p in sources]
            and input_ids == [data_identity(p) for p in inputs],
        "actual_original_functions_bindings_and_argv_restored": all(v for k, v in bindings.items() if k != "new_optimization_count")
            and grid["original_binding_restored"] and support["original_main_restored"],
        "private30GiB_free20GiB": sum(p.stat().st_size for p in BASE.rglob("*") if p.is_file()) <= 30 * 1024**3
            and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
    }
    close_seconds = time.perf_counter() - close_start
    phase = setup_seconds + sum(r["seconds"] for r in calls) + restore_seconds + close_seconds
    result = {"status": "complete_full1050_original_export_vector_guarded" if all(checks.values()) else "failed_full1050_guard",
        "mode": args.mode, "checks": checks, "sampling_calls": calls, "support_trace": support,
        "source_input_setup_seconds": setup_seconds, "binding_restore_seconds": restore_seconds, "close_verify_seconds": close_seconds,
        "complete_affected_phase_seconds": phase, "original_main_seconds": main_seconds,
        "whole_before_result_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "phase_seconds": phase,
                      "whole_before_result_seconds": result["whole_before_result_seconds"]}), flush=True)
    if not all(checks.values()): raise AssertionError("A full1050 original export protection failed")


if __name__ == "__main__":
    main()
