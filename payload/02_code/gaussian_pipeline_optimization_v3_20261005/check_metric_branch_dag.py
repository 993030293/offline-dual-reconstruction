"""Check isolated original metric branches on native heterogeneous fixtures."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

STARTED = time.perf_counter()
from metric_branch_dag import COMMON, LABELS, ORIGINAL, V1, prepare_leaf
import numpy as np
import pycolmap
from runtime import atomic_json, data_identity
from profile_direct_schur import same_model


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close_geometry(left, right):
    exact = same_model(left, right)
    checks = {k: v for k, v in exact.items() if k not in (
        "all_images_full_xy_assignments_and_pose", "all_xyz_color_error_tracks", "all_frame_poses_and_rig_assignment")}
    if not all(checks.values()):
        return False, exact, None
    maximum = 0.
    for key, image in left.images.items():
        other = right.images[key]
        if (image.name != other.name or image.camera_id != other.camera_id or image.frame_id != other.frame_id
                or image.has_pose != other.has_pose or len(image.points2D) != len(other.points2D)
                or any(a.point3D_id != b.point3D_id or not np.array_equal(a.xy, b.xy)
                       for a, b in zip(image.points2D, other.points2D))):
            return False, exact, None
        if image.has_pose:
            maximum = max(maximum, float(np.max(np.abs(image.cam_from_world().matrix()-other.cam_from_world().matrix()))))
    for key, point in left.points3D.items():
        other = right.points3D[key]
        if (not np.array_equal(point.color, other.color)
                or [(e.image_id, e.point2D_idx) for e in point.track.elements]
                != [(e.image_id, e.point2D_idx) for e in other.track.elements]):
            return False, exact, None
        maximum = max(maximum, float(np.max(np.abs(point.xyz-other.xyz))), abs(point.error-other.error))
    for key, frame in left.frames.items():
        other = right.frames[key]
        if frame.rig_id != other.rig_id or frame.has_pose() != other.has_pose():
            return False, exact, None
        if frame.has_pose():
            maximum = max(maximum, float(np.max(np.abs(frame.rig_from_world.matrix()-other.rig_from_world.matrix()))))
    return maximum <= 1e-10, exact, maximum


def run(mode, model, initial, output, log):
    command = [sys.executable, "-B", str(Path(__file__).with_name("metric_branch_dag.py")),
               "--mode", mode, "--", "--model", str(model), "--initial-frames", str(initial),
               "--output", str(output), "--threads", "8"]
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    if process.returncode:
        raise RuntimeError("Original metric fixture branch failed; see owned log: " + str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("metric_branch_dag.py"), ORIGINAL,
               V1 / "runtime.py", V1 / "pose_gap_candidates.py", COMMON / "common.py",
               COMMON / "run_prior_ba.py", Path(__file__).with_name("profile_direct_schur.py")]
    identities = [data_identity(p) for p in sources]
    atomic_json(args.output / "CONTRACT.json", {"sources": identities, "fixtures": "native irregular single/multi-camera rigs at two scales",
        "finite_final_geometry_absolute_tolerance": 1e-10, "initial_model_and_discrete_metadata": "complete exact",
        "original_solver_options_unchanged": True, "no_real_GT_or_training_claim": True, "counted": False})
    checks = []
    for index, (scale, camera_model, cameras) in enumerate(((1., "SIMPLE_RADIAL", 1), (3., "PINHOLE", 2))):
        for label in LABELS:
            prepare_leaf(label)
        pycolmap.set_random_seed(119 + index)
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_frames_per_rig = 2, 6
        options.num_cameras_per_rig, options.num_points3D = cameras, 240
        options.camera_model_id = getattr(pycolmap.CameraModelId, camera_model)
        options.camera_params = [1280., 512., 384., .05] if cameras == 1 else [1100., 1180., 512., 384.]
        model = pycolmap.synthesize_dataset(options)
        model.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(), np.array([2., -.7, 1.3])))
        rows = [{"image_id": im.image_id, "image_name": im.name, "C_W_m": im.projection_center().tolist(),
                 "position_covariance_m2": (np.eye(3)*.15**2).tolist()} for im in model.images.values() if im.has_pose]
        rng = np.random.default_rng(119 + index)
        for point in model.points3D.values():
            point.xyz = point.xyz + rng.normal(0., .002*scale, 3)
        folder = args.output / ("fixture_" + str(index))
        folder.mkdir()
        model_path, initial = folder / "input_model", folder / "initial"
        model_path.mkdir(); initial.mkdir()
        model.write(model_path)
        atomic_json(initial / "INITIAL_FRAME_MANIFEST.json", {"selected_frames": rows, "synthetic_not_real_coarse_GT": True})
        for mode in ("serial", "dag"):
            run(mode, model_path, initial, folder / mode, folder / (mode + ".log"))
        serial, dag = read(folder / "serial/RESULT.json"), read(folder / "dag/RESULT.json")
        case = {"fixture": index, "selected_equal": serial["selected"] == dag["selected"],
                "original_P95_guard": serial["development_guard_passed"] and dag["development_guard_passed"], "branches": {}}
        for label in LABELS:
            control = folder / "serial"
            candidate = folder / "dag/branches" / label
            left = pycolmap.Reconstruction(serial["results"][label]["model"])
            right = pycolmap.Reconstruction(dag["results"][label]["model"])
            equivalent, exact, difference = close_geometry(left, right)
            binary_names = {"cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"}
            control_initial = control / (label+"_INITIAL_MODEL")
            candidate_initial = candidate / (label+"_INITIAL_MODEL")
            initial_equal = ({p.name for p in control_initial.iterdir()} == binary_names
                and {p.name for p in candidate_initial.iterdir()} == binary_names
                and all((control_initial / name).read_bytes() == (candidate_initial / name).read_bytes() for name in binary_names))
            case["branches"][label] = {
                "complete_initial_model_bytes_equal": initial_equal,
                "effective_options_equal": read(control / (label+"_EFFECTIVE_OPTIONS.json")) == read(candidate / (label+"_EFFECTIVE_OPTIONS.json")),
                "complete_final_model_equivalent": equivalent, "complete_exact_fields": exact,
                "maximum_final_geometry_absolute_difference": difference,
                "original_factor_and_observation_guards": serial["results"][label]["factor_count_verified"]
                    and dag["results"][label]["factor_count_verified"] and dag["results"][label]["common_observations_preserved"]}
        close = read(folder / "dag/DAG_RESOURCE_CLOSE.json")
        intervals = close["branch_intervals"]
        case["actual_branch_overlap_seconds"] = min(v["ended_monotonic"] for v in intervals.values()) - max(v["started_monotonic"] for v in intervals.values())
        case["normal_owned_close"] = not close["alive_owned_processes"] and close["cancellation_kills"] == 0
        checks.append(case)
        passed = case["selected_equal"] and case["original_P95_guard"] and case["normal_owned_close"] and case["actual_branch_overlap_seconds"] > 0
        passed = passed and all(all(branch[key] for key in ("complete_initial_model_bytes_equal", "effective_options_equal",
            "complete_final_model_equivalent", "original_factor_and_observation_guards")) for branch in case["branches"].values())
        atomic_json(args.output / "RESULT.json", {"status": "fixture_passed_next_pending" if passed else "required_generic_guard_failed",
            "completed": checks, "elapsed_seconds": time.perf_counter()-STARTED, "counted": False})
        if not passed:
            raise AssertionError("The preregistered complete generic oracle failed")
    if identities != [data_identity(p) for p in sources]:
        raise AssertionError("Sources changed during the generic oracle")
    atomic_json(args.output / "RESULT.json", {"status": "two_heterogeneous_native_fixtures_passed_lifecycle_and_cost_pending",
        "completed": checks, "elapsed_seconds": time.perf_counter()-STARTED, "sources_unchanged": True, "counted": False})


if __name__ == "__main__":
    main()
