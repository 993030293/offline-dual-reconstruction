"""Check native geometry, deregistration, mutation, interruption and fresh recovery."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pycolmap

from check_grid_support import ROOT, POLICY, SOURCE, source_module
from grid_support import grid_support_execution
from machine_json import exact_json_values
import segmented_camera_support as owned
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001")
    args.output.mkdir(parents=True, exist_ok=False)
    folders = sorted(path for path in (ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/generic_binary_qa_001").iterdir()
                     if path.is_dir() and (path/"images.bin").is_file())
    if len(folders) != 9:
        raise ValueError("All nine existing native geometry and calibration fixtures are required")
    inputs = [data_identity(path) for folder in folders for path in sorted(folder.glob("*.bin"))]
    paths = [Path(__file__), Path(owned.__file__), Path(__file__).with_name("native_observation_index.py"), SOURCE, POLICY]
    sources = [data_identity(path) for path in paths]
    module = source_module()
    original_metric = module.camera_geometry_metrics
    policy = json.loads(POLICY.read_text(encoding="utf-8"))["camera_geometry_admission"]
    original_rows, original_index = owned.segmented_rows, owned.SupportObservationIndex
    created = []

    class ObservedIndex(original_index):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    checks, records = {}, []
    try:
        owned.SupportObservationIndex = ObservedIndex
        with grid_support_execution(module, True):
            for ordinal, folder in enumerate(folders):
                model = pycolmap.Reconstruction(folder)
                images = [image for image in model.images.values() if image.has_pose]
                expected = [module.camera_geometry_metrics(model, image, policy) for image in images]
                actual, trace = owned.native_support_many(model, images, policy, args.output/f"native{ordinal}", module.camera_geometry_metrics)
                exact_json_values(expected, actual)
                if trace["route"] != "whole_map_segmented_support" or not trace["index"]["omitted_image_metadata_pixels_and_bindings_exact"]:
                    raise AssertionError("The supported native fixture did not execute its verified segmented route")
                checks[f"native_complete_values_types_order_bits_{ordinal}"] = True
                records.append({"fixture": folder.name, "images": len(images), "trace": trace})
            model = pycolmap.Reconstruction(folders[0])
            image = next(image for image in model.images.values() if image.has_pose)
            model.deregister_frame(image.frame_id)
            images = [image for image in model.images.values() if image.has_pose]
            expected = [module.camera_geometry_metrics(model, image, policy) for image in images]
            actual, trace = owned.native_support_many(model, images, policy, args.output/"actual_registration_regression", module.camera_geometry_metrics)
            exact_json_values(expected, actual)
            checks["same_actual_deregistered_failure_repaired_complete_values"] = trace["route"] == "whole_map_segmented_support"
            checks["unregistered_image_fields_verified_with_registered_rows"] = trace["index"]["unregistered_images"] > 0
            for exception in (RuntimeError, KeyboardInterrupt, SystemExit):
                def interrupted(*args, **kwargs):
                    raise exception("owned segmented calculation interrupted")
                owned.segmented_rows = interrupted
                caught = None
                try:
                    owned.native_support_many(model, images, policy, args.output/exception.__name__, module.camera_geometry_metrics)
                except BaseException as error:
                    caught = error
                finally:
                    owned.segmented_rows = original_rows
                if type(caught) is not exception or str(caught) != "owned segmented calculation interrupted" or not created[-1].verified:
                    raise AssertionError("A segmented interruption changed its original exception or skipped cleanup")
                checks[exception.__name__+"_propagated_and_owned_state_verified"] = True
            for mutation in ("registered_pixels", "omitted_pixels", "point_coordinates", "calibration"):
                current = pycolmap.Reconstruction(model)
                registered = [image for image in current.images.values() if image.has_pose]
                def mutate(*args, **kwargs):
                    value = original_rows(*args, **kwargs)
                    if mutation == "point_coordinates":
                        next(iter(current.points3D.values())).xyz += np.array([.01, 0., 0.])
                    elif mutation == "calibration":
                        next(iter(current.cameras.values())).params[0] += .5
                    else:
                        target = next(image for image in current.images.values()
                                      if image.has_pose == (mutation == "registered_pixels"))
                        target.points2D[0].xy += np.array([.125, 0.])
                    return value
                owned.segmented_rows = mutate
                caught = None
                try:
                    owned.native_support_many(current, registered, policy, args.output/mutation, module.camera_geometry_metrics)
                except ValueError as error:
                    caught = str(error)
                finally:
                    owned.segmented_rows = original_rows
                if caught is None or created[-1].verified:
                    raise AssertionError("A private map mutation escaped its complete registered or omitted state guard")
                checks[mutation+"_rejected_before_result"] = True
            blocked = args.output/"existing_before_directory"
            (blocked/"before").mkdir(parents=True)
            caught = None
            try:
                owned.native_support_many(model, images, policy, blocked, module.camera_geometry_metrics)
            except FileExistsError as error:
                caught = error
            if caught is None:
                raise AssertionError("The actual owned snapshot write error was swallowed")
            checks["actual_snapshot_write_failure_propagated"] = True
            actual, trace = owned.native_support_many(model, images, policy, args.output/"fresh_directory_recovery", module.camera_geometry_metrics)
            exact_json_values(expected, actual)
            checks["fresh_directory_recovery_complete_original_values"] = trace["closed"] and trace["index"]["complete_native_state_unchanged"]
        checks["original_source_bindings_restored"] = module.camera_geometry_metrics is original_metric
        checks["all_original_inputs_and_sources_unchanged"] = inputs == [data_identity(path) for folder in folders for path in sorted(folder.glob("*.bin"))]
        checks["current_sources_unchanged"] = sources == [data_identity(path) for path in paths]
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"status": "failed_native_segmented_lifecycle", "error": repr(error),
            "checks_before_failure": checks, "sources": sources, "inputs": inputs,
            "complete_seconds": time.perf_counter()-started, "new_optimization_count": 0})
        raise
    finally:
        owned.segmented_rows, owned.SupportObservationIndex = original_rows, original_index
    checks["private_index_and_kernel_bindings_restored"] = owned.segmented_rows is original_rows and owned.SupportObservationIndex is original_index
    result = {"status": "native_segmented_support_lifecycle_passed" if all(checks.values()) else "failed_native_lifecycle_guard",
        "checks": checks, "records": records, "sources": sources, "inputs": inputs,
        "complete_seconds": time.perf_counter()-started, "new_optimization_count": 0,
        "full_export_cost_and_nonbox_external": "pending"}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks}))
    if not all(checks.values()):
        raise AssertionError("A required actual native lifecycle protection failed")


if __name__ == "__main__":
    main()
