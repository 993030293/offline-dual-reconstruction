"""Compare the fixed filter on the entire existing development native model."""

import time

STARTED = time.perf_counter()

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import pycolmap

from check_point_filter_delete_prefix import structured
from check_point_filter_predicate import outcome
from point_filter_oracles import compare_outputs, exact_values
from point_filter_predicate import ROOT, V1, native, point_filter_execution
from profile_point_filter_predicate import original_metric_module
from runtime import atomic_json, data_identity


OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pf_v4_001"
MODEL = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/sparse/0"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native_count(value):
    """Read the installed binding's method or property without dropping a field."""
    return int(value() if callable(value) else value)


def snapshot(model, folder):
    """Retain all serialized fields plus live order and unwritten cached fields."""
    folder.mkdir(parents=True, exist_ok=False)
    writing = outcome(lambda: model.write_binary(folder))
    if not writing["ok"]:
        raise AssertionError("The valid external model native writer failed: " + repr(writing))
    members = [path.name for path in sorted(folder.iterdir()) if path.is_file()]
    if members != ["cameras.bin", "frames.bin", "images.bin", "points3D.bin", "rigs.bin"]:
        raise AssertionError("The native external writer changed its complete membership")
    queries = ("num_cameras", "num_frames", "num_images", "num_points3D", "num_reg_frames",
               "num_reg_images", "num_rigs", "compute_num_observations", "compute_mean_track_length",
               "compute_mean_reprojection_error", "compute_mean_observations_per_reg_image")
    images = []
    for key, image in model.images.items():
        images.append((int(key), image.name, int(image.camera_id), int(image.frame_id), bool(image.has_pose),
                       bool(image.has_camera_ptr()), bool(image.has_frame_ptr()),
                       native_count(image.num_points2D), native_count(image.num_points3D),
                       image.cam_from_world().matrix() if image.has_pose else None))
    cameras = [(int(key), str(camera.model), int(camera.width), int(camera.height),
                camera.params, bool(camera.has_prior_focal_length)) for key, camera in model.cameras.items()]
    # The full native binary comparison also checks every track and point2D slot.
    state = {"ordered_point_ids": np.fromiter(model.points3D, dtype=np.uint64, count=model.num_points3D()),
             "ordered_track_lengths": np.fromiter((point.track.length() for point in model.points3D.values()),
                                                   dtype=np.int64, count=model.num_points3D()),
             "images": images, "cameras": cameras,
             "rig_fields": [(int(key), value.todict()) for key, value in model.rigs.items()],
             "frame_fields": [(int(key), value.todict()) for key, value in model.frames.items()],
             "native_aggregate_queries": [(name, outcome(getattr(model, name))) for name in queries]}
    return {"live_fields": structured(state), "native_binary_write": writing, "members": members}


def check_bounds(output):
    if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > 30 * 1024**3:
        raise MemoryError("The independent external output budget was exceeded")
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise OSError("The external component disk headroom was exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    module, common = original_metric_module()
    policy_path = common.PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json"
    policy = read(policy_path)
    sources = [Path(__file__), Path(__file__).with_name("point_filter_predicate.py"),
               Path(__file__).with_name("point_filter_oracles.py"),
               Path(__file__).with_name("check_point_filter_delete_prefix.py"),
               Path(__file__).with_name("check_point_filter_predicate.py"),
               Path(__file__).with_name("profile_point_filter_predicate.py"), Path(native.__file__),
               Path(module.__file__), V1 / "runtime.py", policy_path, Path(pycolmap._core.__file__)]
    paths = sources + sorted(MODEL.glob("*.bin"))
    identities = [data_identity(path) for path in paths]
    copies = output / "executed_sources"
    copies.mkdir()
    for index, path in enumerate(sources):
        if path.suffix in (".py", ".json"):
            shutil.copyfile(path, copies / (str(index) + "_" + path.name))
    atomic_json(output / "CONTRACT.json", {
        "sources_and_inputs": identities, "expected_initial_denominator": [360, 360, 133020, 848219],
        "policy_unchanged": True, "R111_R112_disabled": True, "paired_cost_is_diagnostic_only": True,
        "existing_Oxford_development_not_blind_or_metric_GT": True,
        "additional_frame_stress_rule": "smallest_registered_frame_ID_after_two_original_filter_passes",
        "new_optimization_count": 0})
    records, models, states = [], [], []
    label, stage = "not_started", "prepare"
    try:
        for enabled in (False, True):
            label = "columns" if enabled else "native"
            start = time.perf_counter()
            model = pycolmap.Reconstruction(MODEL)
            denominator = [model.num_images(), model.num_reg_images(), model.num_points3D(), model.compute_num_observations()]
            if denominator != [360, 360, 133020, 848219]:
                raise AssertionError("The unchanged complete external denominator differed")
            centers = np.asarray([image.projection_center() for image in model.images.values() if image.has_pose])
            if not np.isfinite(centers).all():
                raise AssertionError("The real external camera positions are nonfinite")
            point_policy = policy["point_filter"]
            lo = centers.min(0) - point_policy["camera_aabb_margin_m"]
            hi = centers.max(0) + point_policy["camera_aabb_margin_m"]
            side_states = {}
            for step in ("before",):
                stage = step
                side_states[step] = snapshot(model, output / label / step)
            stage = "original_two_pass_filter_and_frame_admission"
            with point_filter_execution(native, enabled) as trace:
                removed_before = native.filter_points(model, point_policy, lo, hi)
                side_states["after_first"] = snapshot(model, output / label / "after_first")
                support_before = {image.name: module.camera_geometry_metrics(model, image, policy["camera_geometry_admission"])
                                  for image in model.images.values() if image.has_pose}
                excluded = {name for name, row in support_before.items() if not row["passed"]}
                kept = set(support_before) - excluded
                frames = {image.frame_id for image in model.images.values() if image.has_pose and image.name in excluded}
                if any(image.has_pose and image.frame_id in frames and image.name in kept for image in model.images.values()):
                    raise AssertionError("The original shared rig frame guard failed")
                for frame_id in sorted(frames):
                    model.deregister_frame(frame_id)
                side_states["after_admission_frames"] = snapshot(model, output / label / "after_admission_frames")
                removed_after = native.filter_points(model, point_policy, lo, hi)
                support_after = {image.name: module.camera_geometry_metrics(model, image, policy["camera_geometry_admission"])
                                 for image in model.images.values() if image.has_pose}
                side_states["after_second"] = snapshot(model, output / label / "after_second")
                registered_frames = sorted({image.frame_id for image in model.images.values() if image.has_pose})
                if not registered_frames:
                    raise AssertionError("No actual native frame remained for the registered mutation guard")
                forced_frame = registered_frames[0]
                registered_before_stress = model.num_reg_images()
                model.deregister_frame(forced_frame)
                if model.num_reg_images() >= registered_before_stress:
                    raise AssertionError("The fixed frame stress branch did not deregister a real image")
                side_states["after_forced_frame"] = snapshot(model, output / label / "after_forced_frame")
                removed_stress = native.filter_points(model, point_policy, lo, hi)
                side_states["after_stress_filter"] = snapshot(model, output / label / "after_stress_filter")
            if not trace["original_binding_restored"] or len(trace["calls"]) != 3:
                raise AssertionError("The full external original binding or call count changed")
            if enabled and any(row["route"] != "columns" for row in trace["calls"]):
                raise AssertionError("The admitted real external model silently used an original route")
            scientific = {"initial_denominator": denominator, "lo": structured(lo), "hi": structured(hi),
                          "removed_before": removed_before, "removed_after": removed_after,
                          "support_before": support_before, "support_after": support_after,
                          "excluded_names": sorted(excluded), "deregistered_admission_frames": sorted(frames),
                          "forced_stress_frame": int(forced_frame), "removed_after_stress": removed_stress,
                          "final_denominator": [model.num_images(), model.num_reg_images(), model.num_points3D(), model.compute_num_observations()],
                          "camera_radius_in_unverified_model_units": float(np.linalg.norm(centers, axis=1).max())}
            atomic_json(output / label / "SCIENTIFIC.json", scientific)
            atomic_json(output / label / "LIVE_STATES.json", side_states)
            if [data_identity(path) for path in paths] != identities:
                raise AssertionError("The external source or complete original inputs changed")
            check_bounds(output)
            records.append({"label": label, "scientific": scientific, "trace": trace,
                            "paired_full_component_seconds": time.perf_counter() - start})
            models.append(model)
            states.append(side_states)
            atomic_json(output / "STATE.json", {"status": "actual_complete_side_done", "completed_sides": len(records),
                                                "new_optimization_count": 0})
        stage = "complete_actual_comparison"
        exact_values(records[0]["scientific"], records[1]["scientific"])
        exact_values(*states)
        comparisons = {step: compare_outputs(output / "native" / step, output / "columns" / step) for step in states[0]}
        if any(row["complete_byte_pairs"] != 5 or row["complete_JSON_pairs"] != 0 for row in comparisons.values()):
            raise AssertionError("The complete external native binary comparison changed membership")
        if [data_identity(path) for path in paths] != identities:
            raise AssertionError("The full external inputs changed at actual close")
        check_bounds(output)
        scientific = records[0]["scientific"]
        result = {"status": "complete_external_native_filter_component_exact", "source_inputs_unchanged": True,
                  "comparisons": comparisons, "paired_rows": records,
                  "full_native_fields_and_associations_byte_exact": True, "all_live_order_cache_pose_rig_frame_and_queries_exact": True,
                  "actual_fixed_frame_deregister_and_live_column_rebuild": True,
                  "two_filter_passes_and_original_camera_admission": True,
                  "original_admission_frames_deregistered": len(scientific["deregistered_admission_frames"]),
                  "additional_stress_frame_deregistered": 1,
                  "original_policy_and_no_object_shape_prior": True, "new_optimization_count": 0,
                  "existing_Oxford_development_not_metric_or_training_admission_or_blind_GT": True,
                  "complete_prepublication_seconds": time.perf_counter() - STARTED}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "states": len(comparisons),
                          "initial_denominator": scientific["initial_denominator"],
                          "admission_frames": result["original_admission_frames_deregistered"],
                          "final_denominator": scientific["final_denominator"]}))
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"label": label, "stage": stage, "error_type": type(error).__name__,
                                               "error": str(error), "completed_sides": len(records),
                                               "new_optimization_count": 0, "old_failure_preserved": True})
        raise
    finally:
        models.clear()
        gc.collect()


if __name__ == "__main__":
    main()
