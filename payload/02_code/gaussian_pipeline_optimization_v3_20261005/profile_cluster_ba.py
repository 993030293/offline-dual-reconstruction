"""Run a fresh complete original metric node with an isolated solver selector."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
import numpy as np
import pycolmap
from cluster_ba import COMMON, ROOT, V1, CandidateGuardFailure, differences, metric_binding, plain
import common
import run_metric_refinement as original
from runtime import atomic_json, data_identity

V1OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005"
MODEL = V1OUT / "p23_track_completion_001/metric_filtered_model"
INITIAL = V1OUT / "p1_raw_combined_001"
GAPS = V1OUT / "p7_pose_gap_recovery_001/POSE_GAP_DECISIONS.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_paths():
    return [Path(__file__), Path(__file__).with_name("cluster_ba.py"), Path(__file__).with_name("direct_schur.py"),
            Path(__file__).with_name("cluster_ba_node.py"), Path(__file__).with_name("cluster_ba_node_config.py"),
            Path(__file__).with_name("run_bounded_journal_task.py"),
            Path(__file__).with_name("check_cluster_ba.py"), V1 / "run_metric_refinement.py",
            V1 / "runtime.py", V1 / "pose_gap_candidates.py", COMMON / "common.py",
            COMMON / "run_prior_ba.py", V1 / "run_bounded_task.py",
            Path(__file__).with_name("run_with_resource_lease.py"), Path(pycolmap._core.__file__)]


def input_paths():
    return [MODEL / name for name in ("cameras.bin", "images.bin", "points3D.bin")] + [
            INITIAL / "INITIAL_FRAME_MANIFEST.json", INITIAL / "frame_decisions.jsonl", GAPS]


def same_model(left, right):
    checks = {"valid": right.is_valid(), "camera_keys": set(left.cameras) == set(right.cameras),
              "image_keys": set(left.images) == set(right.images),
              "point_keys": set(left.points3D) == set(right.points3D),
              "frame_keys": set(left.frames) == set(right.frames), "rig_keys": set(left.rigs) == set(right.rigs)}
    if not all(checks.values()): return checks
    checks["all_camera_parameters"] = all(
        camera.model == right.cameras[key].model and camera.width == right.cameras[key].width
        and camera.height == right.cameras[key].height and np.array_equal(camera.params, right.cameras[key].params)
        for key, camera in left.cameras.items())
    checks["all_images_full_xy_assignments_and_pose"] = True
    for key, image in left.images.items():
        other = right.images[key]
        if (image.name != other.name or image.camera_id != other.camera_id or image.frame_id != other.frame_id
                or image.has_pose != other.has_pose or len(image.points2D) != len(other.points2D)
                or (image.has_pose and not np.array_equal(image.cam_from_world().matrix(), other.cam_from_world().matrix()))):
            checks["all_images_full_xy_assignments_and_pose"] = False; break
        if any(a.point3D_id != b.point3D_id or not np.array_equal(a.xy, b.xy)
               for a, b in zip(image.points2D, other.points2D)):
            checks["all_images_full_xy_assignments_and_pose"] = False; break
    checks["all_xyz_color_error_tracks"] = True
    for key, point in left.points3D.items():
        other = right.points3D[key]
        if (not np.array_equal(point.xyz, other.xyz) or not np.array_equal(point.color, other.color)
                or point.error != other.error or
                [(e.image_id, e.point2D_idx) for e in point.track.elements] !=
                [(e.image_id, e.point2D_idx) for e in other.track.elements]):
            checks["all_xyz_color_error_tracks"] = False; break
    checks["all_frame_poses_and_rig_assignment"] = all(
        frame.rig_id == right.frames[key].rig_id and frame.has_pose() == right.frames[key].has_pose()
        and (not frame.has_pose() or np.array_equal(frame.rig_from_world.matrix(), right.frames[key].rig_from_world.matrix()))
        for key, frame in left.frames.items())
    checks["all_sensor_from_rig_poses"] = True
    for key, rig in left.rigs.items():
        other = right.rigs[key]
        if rig.ref_sensor_id != other.ref_sensor_id or set(rig.sensor_ids()) != set(other.sensor_ids()):
            checks["all_sensor_from_rig_poses"] = False; break
        for sensor in rig.non_ref_sensors:
            if (rig.has_sensor_from_rig(sensor) != other.has_sensor_from_rig(sensor)
                    or (rig.has_sensor_from_rig(sensor) and not np.array_equal(rig.sensor_from_rig(sensor).matrix(), other.sensor_from_rig(sensor).matrix()))):
                checks["all_sensor_from_rig_poses"] = False; break
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["auto", "cluster"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    phases, captures = {}, []
    prior_argv = sys.argv
    publish_original = original.atomic_json
    factories = (pycolmap.create_default_ceres_bundle_adjuster, pycolmap.create_pose_prior_ceres_bundle_adjuster)
    phase, failure, result = "source_input_preparation", None, None
    sources, inputs = [], []
    try:
        prepare = time.perf_counter()
        sources, inputs = [data_identity(path) for path in source_paths()], [data_identity(path) for path in input_paths()]
        reference = pycolmap.Reconstruction(MODEL)
        identity = original.observation_identity(reference)
        if reference.num_reg_images() != 1021 or reference.num_points3D() != 75846 or len(identity) != 959881:
            raise CandidateGuardFailure("Frozen complete input counts changed")
        intrinsics = {key: camera.params.copy() for key, camera in reference.cameras.items()}
        atomic_json(args.output / "CONTRACT.json", {"mode": args.mode, "source": sources, "inputs": inputs,
                   "binding": {"version": pycolmap.__version__, "COLMAP_version": pycolmap._core.COLMAP_version,
                               "COLMAP_build": pycolmap._core.COLMAP_build, "Ceres_version": pycolmap._core.__ceres_version__,
                               "has_cuda": pycolmap.has_cuda},
                   "input_counts": {"cameras": reference.num_cameras(), "registered_images": reference.num_reg_images(),
                                    "points3D": reference.num_points3D(), "observations": len(identity)},
                   "no_GT_or_old_metric_transform": True, "unchanged_original_pipeline": str(V1 / "run_metric_refinement.py"),
                   "native_time_cap_step_boundary_allowed": True, "counted": False})
        phases["source_input_preparation"] = time.perf_counter() - prepare

        def capture(options, config, local, priors=None, prior_options=None):
            expected = common_original_options(60)
            expected.ceres.solver_options.max_solver_time_in_seconds = 180
            expected.ceres.solver_options.num_threads = 8
            if args.mode == "cluster":
                expected.ceres.auto_select_solver_type = False
                expected.ceres.solver_options.linear_solver_type = type(expected.ceres.solver_options.linear_solver_type).ITERATIVE_SCHUR
                expected.ceres.solver_options.preconditioner_type = type(expected.ceres.solver_options.preconditioner_type).CLUSTER_JACOBI
            diff = differences(plain(expected.todict()), plain(options.todict()))
            label = "visual_metric_aligned" if priors is None else "weak_position_prior"
            record = {"case": label, "effective_options": plain(options.todict()), "unexpected_option_differences": diff,
                      "configured_images": len(config.images), "visual_residuals": int(config.num_residuals(local)),
                      "gauge": config.fixed_gauge.name, "prior_count": 0 if priors is None else len(priors)}
            if prior_options is not None: record["prior_options"] = plain(prior_options.todict())
            atomic_json(args.output / (label + "_EFFECTIVE_OPTIONS.json"), record)
            if diff or len(config.images) != 1021 or config.num_residuals(local) != 1919762:
                raise CandidateGuardFailure("Actual adjuster options/config differ from the frozen original interface")
            if priors is None and config.fixed_gauge != pycolmap.BundleAdjustmentGauge.THREE_POINTS:
                raise CandidateGuardFailure("Original THREE_POINTS visual gauge changed")
            if priors is not None:
                if (len(priors) != 1015 or prior_options.prior_position_fallback_stddev != .15
                        or prior_options.ceres.prior_position_loss_function_type != pycolmap.LossFunctionType.CAUCHY
                        or prior_options.ceres.prior_position_loss_scale != 2.7954834829151074
                        or any(not np.array_equal(p.position_covariance, np.eye(3)*.15**2) for p in priors)):
                    raise CandidateGuardFailure("Original weak-prior count/covariance/loss changed")
            captures.append((label, local))

        def visual(options, config, local):
            capture(options, config, local)
            return factories[0](options, config, local)

        def weak(options, prior_options, config, priors, local):
            capture(options, config, local, priors, prior_options)
            return factories[1](options, prior_options, config, priors, local)

        def publish(path, value):
            publish_original(path, value)
            if Path(path).name == "FAILURE.json" and value.get("status") == "failed_projection_validation":
                raise CandidateGuardFailure("First original full projection failure; later case not run")

        common_original_options = common.ba_options
        phase = "original_complete_metric_pipeline"
        run_started = time.perf_counter()
        original.atomic_json = publish
        pycolmap.create_default_ceres_bundle_adjuster, pycolmap.create_pose_prior_ceres_bundle_adjuster = visual, weak
        sys.argv = [str(V1 / "run_metric_refinement.py"), "--model", str(MODEL), "--initial-frames", str(INITIAL),
                    "--output", str(args.output / "native"), "--include-raw-candidates", "--pose-gap-decisions", str(GAPS), "--threads", "8"]
        with metric_binding(args.mode, args.output): original.main()
        phases["original_complete_metric_pipeline"] = time.perf_counter() - run_started
        phase = "full_model_readback_and_original_selection"
        verify_started = time.perf_counter()
        result = read(args.output / "native/RESULT.json")
        contract = read(args.output / "native/CONTRACT.json")
        checks = {"status": result["status"] == "metric_interface_verified",
                  "weak_selected": result["selected"] == "weak_position_prior",
                  "development_guard": result["development_guard_passed"] and result["common_observation_p95_delta_px"] <= .05,
                  "full_input_and_prior_counts": contract["input_camera_count"] == 1021 and contract["supported_position_prior_count"] == 1015
                    and len(contract["pose_gap_images_without_priors"]) == 6,
                  "sigma_and_scale": contract["sigma_m"] == .15 and contract["prior_loss_scale"] == 2.7954834829151074,
                  "both_cases_executed": len(captures) == 2}
        for label, local in captures:
            item = result["results"][label]
            readback = pycolmap.Reconstruction(Path(item["model"]))
            record = same_model(local, readback)
            validity = original.projection_validity(readback)
            expected_priors = 0 if label == "visual_metric_aligned" else 1015
            record.update(positive_finite_all_projections=validity["valid"], all_observations=identity == original.observation_identity(readback),
                          all_intrinsics=all(np.array_equal(camera.params, intrinsics[key]) for key, camera in readback.cameras.items()),
                          full_residuals=item["solver"]["num_residuals"] == 1919762+3*expected_priors,
                          full_residual_blocks=item["solver"]["num_residual_blocks"] == 959881+expected_priors,
                          original_activation=item["factor_count_verified"], original_usable=item["solver"]["usable"],
                          all_observation_metrics=item["metrics"]["observations"] == 959881)
            native = item["solver"]["native_summary"]["ceres_summary"]
            record["original_full_8_threads"] = native["num_threads_given"] == 8 and native["num_threads_used"] == 8
            if args.mode == "auto": record["original_auto_iterative_schur"] = native["linear_solver_type_used"] == "ITERATIVE_SCHUR"
            atomic_json(args.output / (label + "_FULL_READBACK.json"), {"checks": record, "projection_validity": validity})
            checks[label] = all(record.values())
            del readback
        if not all(checks.values()): raise CandidateGuardFailure("Original selection or full model readback guard rejected")
        phases["full_model_readback_and_original_selection"] = time.perf_counter() - verify_started
        atomic_json(args.output / "GUARDS.json", checks)
    except BaseException as error:
        failure = {"status": "candidate_guard_failed" if isinstance(error, CandidateGuardFailure) else "execution_or_checker_error",
                   "mode": args.mode, "phase": phase, "error": repr(error), "counted": False}
        atomic_json(args.output / "FAILURE.json", failure)
    finally:
        restore_started = time.perf_counter()
        sys.argv = prior_argv
        original.atomic_json = publish_original
        pycolmap.create_default_ceres_bundle_adjuster, pycolmap.create_pose_prior_ceres_bundle_adjuster = factories
        restore = {"argv": sys.argv is prior_argv, "original_publish": original.atomic_json is publish_original,
                   "visual_factory": pycolmap.create_default_ceres_bundle_adjuster is factories[0],
                   "prior_factory": pycolmap.create_pose_prior_ceres_bundle_adjuster is factories[1]}
        atomic_json(args.output / "INTERFACE_RESTORE.json", restore)
        phases["interface_restore"] = time.perf_counter() - restore_started
        validate_started = time.perf_counter()
        identity_unchanged = (sources == [data_identity(path) for path in source_paths()] and inputs == [data_identity(path) for path in input_paths()])
        atomic_json(args.output / "IDENTITY_RECHECK.json", {"unchanged": identity_unchanged, "source": sources, "inputs": inputs})
        phases["complete_source_input_recheck"] = time.perf_counter() - validate_started
        captures.clear()
        gc.collect()
        if not all(restore.values()) or not identity_unchanged:
            failure = failure or {"status": "source_or_restore_failed", "mode": args.mode, "phase": "close", "counted": False}
        atomic_json(args.output / "RESULT.json", {"status": "metric_complete_passed" if failure is None else failure["status"],
                   "mode": args.mode, "failure": failure, "phases_seconds": phases, "sources_inputs_unchanged": identity_unchanged,
                   "interface_restored": all(restore.values()), "counted": False, "original_result": result,
                   "completephase_through_owned_restore_close_seconds": time.perf_counter()-STARTED,
                   "launcher_resource_publish_exit_close_measured_by_parent": True})
    if failure is not None: raise SystemExit(1)


if __name__ == "__main__": main()


