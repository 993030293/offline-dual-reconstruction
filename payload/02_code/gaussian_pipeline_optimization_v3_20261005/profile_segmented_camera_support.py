"""Measure every camera on the current map with complete native ownership guards."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil

import psutil
import pycolmap

from check_grid_support import source_module, SOURCE, POLICY, ROOT
from grid_support import grid_support_execution
from native_observation_index import snapshot
from segmented_camera_support import native_support_many
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "segmented"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001")
    args.output.mkdir(parents=True, exist_ok=False)
    export = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003"
    admission = json.loads((export / "TRAINING_ADMISSION.json").read_text(encoding="utf-8"))
    model_path = Path(admission["training_opencv_model"])
    paths = [Path(__file__), Path(__file__).with_name("segmented_camera_support.py"),
             Path(__file__).with_name("native_observation_index.py"), Path(__file__).with_name("binary_map_qa.py"),
             Path(__file__).with_name("grid_support.py"), SOURCE, POLICY]
    input_paths = [export / "TRAINING_ADMISSION.json", *sorted(model_path.glob("*.bin"))]
    sources, inputs = [data_identity(path) for path in paths], [data_identity(path) for path in input_paths]
    saved = args.output / "source_snapshot"
    saved.mkdir()
    for ordinal, source in enumerate(paths):
        shutil.copyfile(source, saved / (f"{ordinal:03d}_" + source.name))
    atomic_json(args.output / "CONTRACT.json", {"sources": sources, "inputs": inputs, "mode": args.mode,
        "common_accepted_R112": True, "complete_model_and_original_five_fields": True, "new_optimization_count": 0})
    module = source_module()
    original = module.camera_geometry_metrics
    policy = json.loads(POLICY.read_text(encoding="utf-8"))["camera_geometry_admission"]
    model = pycolmap.Reconstruction(model_path)
    images = [image for image in model.images.values() if image.has_pose]
    if len(images) != 1016 or model.num_points3D() != 75767 or model.compute_num_observations() != 959120:
        raise ValueError("The current complete map denominator changed")
    before = snapshot(model, args.output / "owner_before")
    setup_seconds = time.perf_counter() - STARTED
    try:
        with grid_support_execution(module, True) as grid_trace:
            started = time.perf_counter()
            if args.mode == "native":
                rows = [module.camera_geometry_metrics(model, image, policy) for image in images]
                trace = {"route": "original_scalar_with_R112", "closed": True,
                         "images": len(images), "complete_seconds": time.perf_counter()-started}
            else:
                rows, trace = native_support_many(model, images, policy, args.output / "batch_state",
                                                 module.camera_geometry_metrics)
            working_seconds = time.perf_counter() - started
        closing = time.perf_counter()
        after = snapshot(model, args.output / "owner_after")
        checks = {
            "all1016_original_five_fields": len(rows) == 1016
                and all(list(row) == ["image_name", "triangulated_points", "occupied_grid_cells", "bbox_area_fraction", "passed"] for row in rows),
            "caller_image_order_kept": [row["image_name"] for row in rows] == [image.name for image in images],
            "complete_native_model_before_after_exact": before == after,
            "actual_registered_admitted_denominators": model.num_reg_images() == 1016
                and model.num_points3D() == 75767 and model.compute_num_observations() == 959120,
            "original_metric_hook_restored": module.camera_geometry_metrics is original and grid_trace["original_binding_restored"],
            "sources_and_inputs_unchanged": sources == [data_identity(path) for path in paths]
                and inputs == [data_identity(path) for path in input_paths],
            "actual_segmented_route_or_original_native": (trace["route"] == "whole_map_segmented_support"
                and trace["index"]["complete_native_state_unchanged"] and trace["index"]["indexed_images"] == 1016)
                if args.mode == "segmented" else len(grid_trace["calls"]) == 1016,
            "complete_helper_closed": trace["closed"],
            "owned_disk_and_headroom": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) <= 30*1024**3
                and shutil.disk_usage(args.output).free >= 20*1024**3,
        }
        closing_seconds = time.perf_counter() - closing
        memory = psutil.Process().memory_info()
        result = {"status": "complete_native_camera_support_diagnostic" if all(checks.values()) else "failed_original_diagnostic_guard",
                  "mode": args.mode, "checks": checks, "sources": sources, "inputs": inputs, "rows": rows,
                  "model_before": before, "model_after": after, "trace": trace,
                  "working_support_seconds": working_seconds, "source_model_owner_setup_seconds": setup_seconds,
                  "complete_close_and_verify_seconds": closing_seconds,
                  "complete_phase_seconds": setup_seconds + grid_trace["setup_seconds"] + working_seconds
                      + grid_trace["restore_seconds"] + closing_seconds,
                  "whole_before_result_seconds": time.perf_counter() - STARTED,
                  "actual_process_peak_RSS_bytes": max(memory.rss, getattr(memory, "peak_wset", memory.rss)),
                  "new_optimization_count": 0, "full_export_original_error_recovery_and_formal_cost": "not_run"}
        atomic_json(args.output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks, "working_seconds": working_seconds,
                          "complete_phase_seconds": result["complete_phase_seconds"]}))
        if not all(checks.values()):
            raise AssertionError("A required complete native support diagnostic guard failed")
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"error": repr(error), "sources": sources, "inputs": inputs,
            "complete_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
