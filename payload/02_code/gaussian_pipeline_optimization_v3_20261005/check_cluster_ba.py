"""Exercise actual visibility cluster solves on fresh native BA fixtures."""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
import numpy as np
import pycolmap
from cluster_ba import COMMON, V1, CandidateGuardFailure, actual_backend, plain, select_cluster
import common
import run_metric_refinement as original
import run_prior_ba as metrics
from runtime import atomic_json, data_identity


def require_error(model, rows, message):
    try:
        common.validate_prior_rows(model, rows)
    except ValueError as error:
        if str(error) != message:
            raise AssertionError("Original first-error precedence changed: " + str(error)) from error
        return message
    raise AssertionError("Original malformed-prior oracle did not reject")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    phase, results, references, errors = "fixture_construction", [], [], {}
    started = time.perf_counter()
    atomic_json(args.output / "CONTRACT.json", {"original_metric_options": "8 threads / 60 iterations / 180 solver seconds / sigma .15 / HUBER1 / CAUCHY2.7954834829151074",
                "changes": ["ceres.auto_select_solver_type=False", "ceres.solver_options.linear_solver_type=ITERATIVE_SCHUR", "ceres.solver_options.preconditioner_type=CLUSTER_JACOBI"],
                "fixtures": "native synthetic irregular 3D points; different scales, camera models and multi-camera rigs; no Box or real GT",
                "preregistered_positive_factor_and_projection_guards": True,
                "no_fallback": True, "counted": False,
                "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("cluster_ba.py")),
                            data_identity(Path(__file__).with_name("direct_schur.py")), data_identity(Path(pycolmap._core.__file__)),
                            data_identity(COMMON / "common.py"), data_identity(V1 / "run_metric_refinement.py")]})
    try:
        for number, (scale, camera_model, cameras) in enumerate([(1., "SIMPLE_RADIAL", 1), (3., "PINHOLE", 2)]):
            phase = "fixture_construction"
            pycolmap.set_random_seed(114 + number)
            synth = pycolmap.SyntheticDatasetOptions()
            synth.num_rigs, synth.num_frames_per_rig = 2, 6
            synth.num_cameras_per_rig, synth.num_points3D = cameras, 240
            synth.camera_model_id = getattr(pycolmap.CameraModelId, camera_model)
            synth.camera_params = [1280., 512., 384., .05] if camera_model == "SIMPLE_RADIAL" else [1100., 1180., 512., 384.]
            model = pycolmap.synthesize_dataset(synth)
            model.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(), np.array([2., -.7, 1.3])))
            references.append({"fixture": number, "scale": scale, "camera_model": camera_model,
                               "num_cameras_per_rig": cameras, "native_valid": model.is_valid()})
            rows = [{"image_id": image.image_id, "image_name": image.name,
                     "C_W_m": image.projection_center().tolist(),
                     "position_covariance_m2": (np.eye(3) * .15**2).tolist()}
                    for image in model.images.values() if image.has_pose]
            common.validate_prior_rows(model, rows)
            errors[str(number)] = {}
            malformed = copy.deepcopy(rows)
            malformed.append(copy.deepcopy(rows[0])); malformed[-1]["C_W_m"] = [float("nan")]*3
            errors[str(number)]["duplicate_before_nonfinite"] = require_error(model, malformed, "Duplicate prior image IDs")
            malformed = copy.deepcopy(rows); malformed[0]["image_id"] = 999999999
            errors[str(number)]["unknown"] = require_error(model, malformed, "Prior refers to an unknown image ID")
            malformed = copy.deepcopy(rows); malformed[0]["image_name"] += "_wrong"
            errors[str(number)]["name"] = require_error(model, malformed, "Prior image identity mismatch")
            malformed = copy.deepcopy(rows); malformed[0]["C_W_m"] = [float("nan")]*3
            errors[str(number)]["finite_position"] = require_error(model, malformed, "Nonfinite prior position")
            malformed = copy.deepcopy(rows); malformed[0]["position_covariance_m2"][0][0] = -1.
            errors[str(number)]["covariance"] = require_error(model, malformed, "Prior covariance must be positive definite")
            malformed = copy.deepcopy(rows); malformed[0]["prior_active"] = True
            for row in malformed[2:]: row["prior_active"] = False
            errors[str(number)]["active_minimum"] = require_error(model, malformed, "At least three active position priors are required")
            malformed = copy.deepcopy(rows)
            for index, row in enumerate(malformed): row["C_W_m"] = [float(index), 0., 0.]
            errors[str(number)]["collinearity"] = require_error(model, malformed, "Active position priors are collinear")
            rng = np.random.default_rng(114 + number)
            for point in model.points3D.values(): point.xyz = point.xyz + rng.normal(0., .002 * scale, 3)
            input_validity = original.projection_validity(model)
            if not model.is_valid() or not input_validity["valid"]:
                raise AssertionError("Fresh synthetic input could not establish a positive-depth valid oracle")
            identity = original.observation_identity(model)
            intrinsics = {key: camera.params.copy() for key, camera in model.cameras.items()}
            fixture = args.output / ("fixture_" + str(number)); fixture.mkdir()
            model.write(fixture)
            atomic_json(fixture / "INPUT_PROJECTION.json", input_validity)
            phase = "candidate_actual_backend_and_contract"
            for label in ("visual_metric_aligned", "weak_position_prior"):
                local = pycolmap.Reconstruction(model)
                config = common.ba_config(local)
                visual = int(config.num_residuals(local))
                options = common.ba_options(60)
                options.ceres.solver_options.max_solver_time_in_seconds = 180
                selection = select_cluster(options)
                if label == "visual_metric_aligned":
                    config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
                    if config.fixed_gauge != pycolmap.BundleAdjustmentGauge.THREE_POINTS:
                        raise CandidateGuardFailure("Original three-point visual gauge was not retained")
                    adjuster = pycolmap.create_default_ceres_bundle_adjuster(options, config, local)
                    prior_count = 0
                else:
                    priors = common.make_priors(local, rows, .15)
                    prior_count = len(priors)
                    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
                    prior_options.prior_position_fallback_stddev = .15
                    prior_options.alignment_ransac.max_error = .6
                    prior_options.alignment_ransac.min_inlier_ratio = .2
                    prior_options.alignment_ransac.random_seed = 0
                    prior_options.alignment_ransac.max_num_trials = 10000
                    prior_options.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.CAUCHY
                    prior_options.ceres.prior_position_loss_scale = 2.7954834829151074
                    adjuster = pycolmap.create_pose_prior_ceres_bundle_adjuster(options, prior_options, config, priors, local)
                case_started = time.perf_counter()
                solution = plain(common.solver_receipt(adjuster.solve()))
                backend = actual_backend(solution)
                result = {"fixture": number, "case": label, "solver": solution, "backend": backend,
                          "selection": selection, "elapsed_seconds": time.perf_counter() - case_started}
                results.append(result)
                atomic_json(args.output / "ACTUAL_SOLVES.json", results)
                if not all(backend.values()): raise CandidateGuardFailure("Actual generic cluster solve rejected its backend guard")
                validity = original.projection_validity(local)
                checks = {"full_residuals": solution["num_residuals"] == visual + 3*prior_count,
                          "full_residual_blocks": solution["num_residual_blocks"] == visual//2 + prior_count,
                          "observations": identity == original.observation_identity(local),
                          "intrinsics": all(np.array_equal(camera.params, intrinsics[key]) for key, camera in local.cameras.items()),
                          "positive_finite_all_projections": validity["valid"],
                          "finite_poses": all(np.isfinite(image.cam_from_world().matrix()).all() for image in local.images.values() if image.has_pose)}
                local.update_point_3d_errors()
                destination = fixture / label; destination.mkdir(); local.write(destination)
                reread = pycolmap.Reconstruction(destination)
                checks["write_readback_all_observations"] = identity == original.observation_identity(reread)
                checks["write_readback_all_xyz"] = all(np.array_equal(point.xyz, reread.points3D[key].xyz) for key, point in local.points3D.items())
                result.update(checks=checks, full_projection=validity, metrics=metrics.observation_metrics(local), active_priors=prior_count)
                atomic_json(args.output / "ACTUAL_SOLVES.json", results)
                if not all(checks.values()): raise CandidateGuardFailure("Actual generic metric residual/gauge/geometry/readback guard failed")
                del adjuster, reread, local
            del model
        phase = "complete"
        gc.collect()
        atomic_json(args.output / "RESULT.json", {"status": "actual_backend_generic_passed", "fixtures": references,
                   "errors": errors, "actual_solve_count": len(results), "counted": False,
                   "setup_and_checks_seconds": time.perf_counter() - started,
                   "whole_including_imports_seconds": time.perf_counter() - STARTED})
    except BaseException as error:
        status = "candidate_guard_failed" if phase == "candidate_actual_backend_and_contract" else "fixture_or_checker_setup_failed"
        atomic_json(args.output / "FAILURE.json", {"status": status, "phase": phase, "error": repr(error),
                    "actual_solve_count": len(results), "counted": False,
                    "elapsed_seconds": time.perf_counter() - STARTED})
        raise


if __name__ == "__main__": main()


