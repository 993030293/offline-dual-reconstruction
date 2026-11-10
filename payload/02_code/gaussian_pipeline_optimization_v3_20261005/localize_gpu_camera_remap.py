"""Run every original development reference and pose with optional exact CUDA pixels."""

import time
STARTED = time.perf_counter()

import argparse
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pycolmap

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
OWNED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from camera_pixel_gpu_remap import camera_pixel_gpu_remap
import integer_sift_matching
import localize_prepared_queries
import localize_new_map_evaluation as native
import metric_branch_localization_003 as observed

EXPORT = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003"
DATABASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p23_track_completion_001/database.db"
INITIAL = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/final_raw_cold_001/initial"
REFERENCE = ROOT / "04_outputs/box2_lio_guided_sfm_v1_20261002T014330"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "CUDA"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    used, free = map(int, subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(","))
    if used > 12 * 1024 or free < 2 * 1024 or shutil.disk_usage(OWNED).free < 20 * 1024**3:
        raise MemoryError("The original GPU and disk headroom are unavailable")
    contract_started = time.perf_counter()
    paths = [Path(__file__), CODE / "camera_pixel_gpu_remap.py", CODE / "camera_remap_stdio_worker.py",
             CODE / "gpu_camera_remap_pixels.py", CODE / "camera_pixel_remap.py", CODE / "mapped_image_linux_guard.py",
             CODE / "camera_remap_matching_guard.py", CODE / "integer_sift_matching.py", CODE / "integer_sift_worker.py",
             CODE / "metric_branch_localization_003.py", CODE / "localize_integer_sift.py",
             CODE / "native_observation_index.py", CODE / "binary_map_qa.py", CODE / "localize_prepared_queries.py",
             CODE / "localize_native_observations.py", CODE / "localize_with_locked_inputs.py",
             CODE / "query_preparation_pipeline.py", CODE / "locked_input_identity.py", V1 / "localize_new_map_evaluation.py",
             V2 / "observation_index.py", V2 / "calibrated_reference_cache.py", V2 / "localize_with_cached_references.py",
             ROOT / "02_code/box2_lio_guided_sfm_v1_20261002/feature_localization.py"]
    source = inspect.getsource(native.main)
    if source.count('pycolmap.Bitmap.from_array(np.asarray(photo.convert("RGB"))), camera)') != 1:
        raise ValueError("The pinned reference caller no longer proves its array-origin metadata scope")
    models = sorted(Path(read(EXPORT / "TRAINING_ADMISSION.json")["training_opencv_model"]).glob("*.bin"))
    split = read(REFERENCE / "00_contract/split_manifest.json")
    images = [REFERENCE / "00_contract/extra_images" / row["image_name"]
              for row in split["extra_frames"] if row["role"] == "H"]
    if len(images) != 24:
        raise ValueError("The original complete reference denominator changed")
    inputs = [DATABASE, EXPORT / "TRAINING_ADMISSION.json", *models, INITIAL / "frame_decisions.jsonl",
              REFERENCE / "00_contract/split_manifest.json", REFERENCE / "00_contract/database.db",
              REFERENCE / "05_evaluation/C_augmented_212/H_LOCALIZATION.json", *images]
    sources, identities = [data_identity(path) for path in paths], [data_identity(path) for path in inputs]
    saved = args.output / "source_snapshot"
    saved.mkdir()
    for ordinal, path in enumerate(paths):
        shutil.copyfile(path, saved / (f"{ordinal:03d}_" + path.name))
    atomic_json(args.output / "CONTRACT.json", {"sources": sources, "inputs": identities, "mode": args.mode,
        "current_R90_R92_R94_R102": True, "cold_private_reference_and_feature_caches": True,
        "complete_all24_reference_preparation_and_original48_matches": True,
        "only_pinned_array_origin_reference_caller_changed": True,
        "common_unchanged_matcher_has_stricter8GiB_Linux_observer_on_both_sides": True,
        "phase_is_complete_single_producer_preparation_plus_owned_lifecycle_contract_and_final_checks": True,
        "new_optimization_count": 0})
    contract_seconds = time.perf_counter() - contract_started
    original_worker, original_wrapper, previous = integer_sift_matching.DistanceWorker, observed.RESOURCE_WRAPPER, sys.argv
    original_undistort, workers = pycolmap.undistort_image, []
    def observe(log):
        worker = observed.ObservedWorker(log)
        workers.append(worker)
        return worker
    try:
        integer_sift_matching.DistanceWorker = observe
        observed.RESOURCE_WRAPPER = CODE / "camera_remap_matching_guard.py"
        with camera_pixel_gpu_remap(args.mode == "CUDA", args.output / "pixels") as trace:
            sys.argv = [str(Path(localize_prepared_queries.__file__)), "--preparation-mode", "parallel",
                "--output", str(args.output / "e"), "--reference-mode", "cache", "--reference-cache-root", str(args.output / "r"),
                "--feature-mode", "cache", "--feature-cache-root", str(args.output / "f"), "--export", str(EXPORT),
                "--database", str(DATABASE), "--initial-frames", str(INITIAL), "--reference-run", str(REFERENCE)]
            localize_prepared_queries.main()
        hits = list((args.output / "e").rglob("H_LOCALIZATION.json"))
        if len(hits) != 1:
            raise ValueError("Exactly one original complete localization result is required")
        result = read(hits[0])
        prep = read(args.output / "e/PREPARATION.json")
        producer = prep["trace"]["producer"]
        index = prep["complete_original_index_trace"]
        guard_started = time.perf_counter()
        gpu = read(args.output / "pixels/WORKER_RESULT.json") if args.mode == "CUDA" else None
        resource = read(args.output / "pixels/WSL_RESOURCE.json") if args.mode == "CUDA" else None
        checks = {"all24_native_poses": result["status"] == "all24_localized_and_frozen" and len(result["rows"]) == 24,
                  "all48_original_matches_and_native_map_verification": len(workers) == 1 and len(workers[0].receipt["calls"]) == 48 and index["calls"] == 48 and index["index"]["complete_native_state_unchanged"],
                  "all24_single_producer_preparations_drained": producer["submitted"] == producer["completed"] == producer["delivered"] == len(producer["preparations"]) == 24 and producer["closed"] and len({row["thread"] for row in producer["preparations"]}) == 1,
                  "all_original_exclusion_and_correspondence_guards": all(row["loc_eval_3D_overlap"] == 0 for row in result["rows"]),
                  "all_sources_and_original_inputs_unchanged": sources == [data_identity(path) for path in paths] and identities == [data_identity(path) for path in inputs],
                  "original_matching_worker_closed_and8GiB_resource_guard": len(workers) == 1 and workers[0].stopped and workers[0].receipt["returncode"] == 0 and read(workers[0].resource)["status"] == "completed" and read(workers[0].resource)["peak_tree_RSS_bytes"] <= 8 * 1024**3,
                  "sampling_and_original_binding_closed": trace["closed"] and trace["original_global_binding_restored"] and pycolmap.undistort_image is original_undistort,
                  "all24_complete_CUDA_calls_or_untouched_native": (len(trace["calls"]) == len(gpu["rows"]) == 24 and trace["map_builds"] == 1 and all(route == "owned_exact_CUDA_array_reference" for route in trace["routes"]) and trace["worker"]["returncode"] == 0 and all(gpu["checks"].values()) and resource["status"] == "completed") if gpu else not trace["routes"],
                  "original_output_and_disk_budget": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) <= 30 * 1024**3 and shutil.disk_usage(OWNED).free >= 20 * 1024**3}
        final_guard_seconds = time.perf_counter() - guard_started
        stage = sum(row["end"] - row["start"] for row in producer["preparations"]) + prep["trace"]["composition_seconds"] + trace["setup_seconds"] + trace["close_seconds"] + contract_seconds + final_guard_seconds
        atomic_json(args.output / "RESULT.json", {"status": "complete_actual_entry_guards_passed" if all(checks.values()) else "failed_verification",
            "mode": args.mode, "checks": checks, "localization": str(hits[0]), "sources": sources, "inputs": identities,
            "trace": trace, "matching_worker": workers[0].receipt if workers else None,
            "preparation": prep, "contract_seconds": contract_seconds, "final_guard_seconds": final_guard_seconds,
            "complete_reference_preparation_stage_seconds": stage, "complete_entry_seconds": time.perf_counter() - STARTED,
            "new_optimization_count": 0, "full1016_training_export_GPU_undistortion": "not_run_outside_array_reference_scope"})
        print(json.dumps({"status": "complete_actual_entry_guards_passed" if all(checks.values()) else "failed_verification", "mode": args.mode, "phase_seconds": stage, "checks": checks}), flush=True)
        if not all(checks.values()):
            raise ValueError("An original complete reference or ownership protection failed")
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"error": repr(error), "sources": sources, "inputs": identities, "new_optimization_count": 0})
        raise
    finally:
        integer_sift_matching.DistanceWorker = original_worker
        observed.RESOURCE_WRAPPER = original_wrapper
        sys.argv = previous
        atomic_json(args.output / "BINDING_RESTORATION.json", {"matching_worker_restored": integer_sift_matching.DistanceWorker is original_worker,
            "matching_resource_wrapper_restored": observed.RESOURCE_WRAPPER is original_wrapper,
            "argv_restored": sys.argv is previous, "original_undistort_binding_restored": pycolmap.undistort_image is original_undistort,
            "new_optimization_count": 0})


if __name__ == "__main__":
    main()
