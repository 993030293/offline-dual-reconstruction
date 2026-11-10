"""Retain the current complete R120/R124 consumer and change only RGB sampling."""

import time
STARTED = time.perf_counter()

import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import shutil
import sys

import psutil
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001"
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005")]
from runtime import atomic_json, data_identity
from native_camera_sampling import native_camera_sampling
import localize_geometry_sift_fusion as consumer
import localize_new_map_evaluation as original_entry


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def owned_consumer(output):
    source_path = CODE/"owned_json_write.py"
    source = source_path.read_text(encoding="utf-8")
    old = 'ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001"'
    new = 'ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001"'
    if source.count(old) != 1:
        raise ValueError("The original bounded owned-write source anchor changed")
    composed = source.replace(old, new)
    namespace = {"__file__": str(source_path), "__name__": "_native_sampling_owned_write"}
    exec(compile(composed, str(source_path), "exec"), namespace)
    original = (consumer.OWNED, consumer.owned_atomic_json, consumer.COMPOSED_SOURCE, sys.argv)
    consumer.OWNED, consumer.owned_atomic_json, consumer.COMPOSED_SOURCE = OWNED, namespace["owned_atomic_json"], composed
    try:
        sys.argv = [str(Path(consumer.__file__)), "--mode", "process", "--output", str(output)]
        yield
    finally:
        consumer.OWNED, consumer.owned_atomic_json, consumer.COMPOSED_SOURCE, sys.argv = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "native_DLL"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.library = args.library.resolve()
    args.output.relative_to(OWNED.resolve())
    args.library.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/"PROCESS_IDENTITY.json", {"pid": psutil.Process().pid,
        "creation_time": psutil.Process().create_time(), "mode": args.mode, "new_optimization_count": 0})
    contract_started = time.perf_counter()
    caller = inspect.getsource(original_entry.main)
    if caller.count('pycolmap.Bitmap.from_array(np.asarray(photo.convert("RGB"))), camera)') != 1:
        raise ValueError("The current complete consumer no longer proves array-origin RGB metadata")
    paths = [Path(__file__), CODE/"native_camera_sampling.py", CODE/"native_camera_sampling.cpp",
             CODE/"camera_pixel_remap.py", CODE/"build_native_camera_sampling.py", args.library,
             Path(consumer.__file__), CODE/"owned_json_write.py"]
    sources = [data_identity(p) for p in paths]
    saved = args.output/"sampling_source_snapshot"
    saved.mkdir()
    for ordinal, path in enumerate(paths):
        shutil.copyfile(path, saved/(f"{ordinal:02d}_"+path.name))
    contract_seconds = time.perf_counter()-contract_started
    original = pycolmap.undistort_image
    original_consumer = (consumer.OWNED, consumer.owned_atomic_json, consumer.COMPOSED_SOURCE, sys.argv)
    restoration = {}
    try:
        with native_camera_sampling(args.mode == "native_DLL", args.library) as trace:
            with owned_consumer(args.output/"u"):
                consumer.main()
        result = read(args.output/"u/RESULT.json")
        prep = read(args.output/"u/e/PREPARATION.json")
        producer = prep["trace"]["producer"]
        guard_started = time.perf_counter()
        checks = {"current_complete_R120_R124_all_guards": all(result["checks"].values()),
                  "both_sides_keep_two_CPU_SIFT_processes": result["mode"] == "process",
                  "all24_complete_preparations_closed": producer["submitted"] == producer["completed"] == producer["delivered"] == len(producer["preparations"]) == 24 and producer["closed"],
                  "sampling_binding_and_owners_restored": trace["closed"] and trace["original_binding_restored"] and pycolmap.undistort_image is original,
                  "consumer_ownership_bindings_restored": (consumer.OWNED, consumer.owned_atomic_json, consumer.COMPOSED_SOURCE, sys.argv) == original_consumer,
                  "all_sampling_sources_unchanged": sources == [data_identity(p) for p in paths],
                  "all24_native_DLL_calls_and_one_map_or_original": (
                      trace["operator"]["sampler_calls"] == 24 and trace["operator"]["builds"] == 1
                      and trace["operator"]["closed"] and trace["operator"]["native_DLL_closed"]
                      and trace["operator"]["routes"] == ["owned_native_calibration_map"]*24
                  ) if args.mode == "native_DLL" else trace["operator"] is None,
                  "original_output_and_disk_budgets": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) <= 30*1024**3 and shutil.disk_usage(OWNED).free >= 20*1024**3}
        final_seconds = time.perf_counter()-guard_started
        stage = (sum(row["end"]-row["start"] for row in producer["preparations"])
                 +prep["trace"]["composition_seconds"]+trace["setup_seconds"]+trace["close_seconds"]
                 +contract_seconds+final_seconds)
        atomic_json(args.output/"RESULT.json", {"status": "complete_actual_entry_guards_passed" if all(checks.values()) else "failed_verification",
            "mode": args.mode, "checks": checks, "localization": result["localization"],
            "sources": sources+result["sources"], "inputs": result["inputs"], "sampling_trace": trace,
            "preparation": prep, "current_consumer_result": result, "contract_seconds": contract_seconds,
            "final_guard_seconds": final_seconds, "complete_reference_preparation_stage_seconds": stage,
            "complete_entry_seconds": time.perf_counter()-STARTED, "new_optimization_count": 0,
            "whole_current1016_training_or_native_Bitmap_read_export": "not_run_outside_registered_reference_scope"})
        print(json.dumps({"mode": args.mode, "phase_seconds": stage, "checks": checks}), flush=True)
        if not all(checks.values()):
            raise ValueError("A complete current native sampling localization guard failed")
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"error": repr(error), "sources": sources, "new_optimization_count": 0})
        raise
    finally:
        restoration.update(sampling_binding_restored=pycolmap.undistort_image is original,
            consumer_bindings_restored=(consumer.OWNED, consumer.owned_atomic_json, consumer.COMPOSED_SOURCE, sys.argv) == original_consumer)
        atomic_json(args.output/"SAMPLING_BINDING_RESTORATION.json", restoration)


if __name__ == "__main__":
    main()
