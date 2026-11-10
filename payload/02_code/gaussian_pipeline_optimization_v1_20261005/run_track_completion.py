"""Complete all registered image tracks before the unchanged metric support gate."""

import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
import pycolmap

from runtime import atomic_json, data_identity
from run_metric_refinement import bind_supported_priors, verified_pose_gap_names, projection_validity


def metric_filter(args, path):
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project/"02_code/box2_lio_guided_sfm_v1_20261002"))
    common = importlib.import_module("common")
    sys.path.insert(0, str(common.PACKAGE/"code/validated_fresh_runtime"))
    from metric_admission import camera_geometry_metrics

    policy_path = common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"
    policy = json.loads(policy_path.read_text())
    model = pycolmap.Reconstruction(path)
    raw = [json.loads(line) for line in (args.initial/"frame_decisions.jsonl").read_text().splitlines()]
    lookup = {row["image_name"]: row for row in raw if row["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")
              and row["coarse_pose_valid"] and not row["excluded_role"]}
    priors, gaps = bind_supported_priors(model, lookup, verified_pose_gap_names(args.initial, args.pose_gap_decisions))
    common.validate_prior_rows(model, priors)
    transform = pycolmap.align_reconstruction_to_locations(model, [row["image_name"] for row in priors],
        np.asarray([row["C_W_m"] for row in priors]), 3,
        pycolmap.RANSACOptions(max_error=.6, min_inlier_ratio=.2, max_num_trials=10000, random_seed=0))
    if transform is None or transform.scale <= 0 or not np.isfinite(transform.scale):
        raise ValueError("The fresh raw-position metric alignment failed")
    model.transform(transform)
    model.update_point_3d_errors()
    atomic_json(args.output/"ALIGNMENT.json", common.serializable({"new_Sim3": transform.todict(),
        "source": "recorded_raw_camera_positions_not_GT", "old_alignment_reused": False,
        "supported_position_count": len(priors), "no_prior_images": gaps}))
    before = [camera_geometry_metrics(model, im, policy["camera_geometry_admission"])
              for im in model.images.values() if im.has_pose]
    centers = np.asarray([im.projection_center() for im in model.images.values() if im.has_pose])
    point_policy = policy["point_filter"]
    if np.linalg.norm(centers, axis=1).max() > point_policy["maximum_camera_radius_m"]:
        raise ValueError("Camera positions exceed the original metric extent policy")
    lo, hi = centers.min(0)-point_policy["camera_aabb_margin_m"], centers.max(0)+point_policy["camera_aabb_margin_m"]
    removed = []
    for point_id in list(model.points3D):
        point = model.points3D[point_id]
        reasons = []
        if not np.isfinite(point.xyz).all():
            reasons.append("nonfinite_xyz")
        if not np.all(point.xyz >= lo) or not np.all(point.xyz <= hi):
            reasons.append("outside_metric_camera_extent")
        if not np.isfinite(point.error) or point.error > point_policy["maximum_point_reprojection_error_px"]:
            reasons.append("reprojection_error")
        if point.track.length() < point_policy["minimum_point_track_length"]:
            reasons.append("track_length")
        if reasons:
            removed.append({"point3D_id": int(point_id), "reasons": reasons,
                            "observations": [[int(el.image_id), int(el.point2D_idx)] for el in point.track.elements]})
            model.delete_point3D(point_id)
    after = [camera_geometry_metrics(model, im, policy["camera_geometry_admission"])
             for im in model.images.values() if im.has_pose]
    validity = projection_validity(model)
    path = args.output/"metric_filtered_model"
    path.mkdir()
    model.write(path)
    atomic_json(args.output/"REMOVED_POINTS.json", removed)
    atomic_json(args.output/"CAMERA_SUPPORT.json", {"before_filter": before, "after_filter": after})
    atomic_json(args.output/"PROJECTION_VALIDITY.json", validity)
    checks = {"all1021_cameras_retained": model.num_reg_images() == 1021,
              "all_cameras_meet_original_support_after_filter": all(row["passed"] for row in after),
              "all_observations_have_valid_projection": validity["valid"],
              "minimum_retained_sparse_points": model.num_points3D() >= point_policy["minimum_retained_sparse_points"]}
    return {"checks": checks, "weak_before_filter": [row for row in before if not row["passed"]],
            "weak_after_filter": [row for row in after if not row["passed"]], "removed_points": len(removed),
            "retained_points": model.num_points3D(), "retained_observations": model.compute_num_observations(),
            "policy": data_identity(policy_path), "model": str(path), "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "images", "initial", "pose-gap-decisions", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.run/"RESULT.json").read_text())
    best = max(source["models"], key=lambda row: row["registered_count"])
    if best["registered_count"] != 1021 or best["original_registered_frames_lost"]:
        raise ValueError("P23 requires the preserved all1021 registered input")
    model_path = Path(best["path"]).resolve()
    database = (args.run/"database.db").resolve()
    source_files = [data_identity(path) for path in sorted(model_path.glob("*.bin"))] + [data_identity(database)]
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 20*1024**3:
        raise OSError("Insufficient disk headroom for the isolated track diagnostic")
    started = time.monotonic()
    binary = Path(__file__).resolve().parents[2]/"05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    state = {"status": "preparing", "pid": psutil.Process().pid, "candidate": "P23_TRACK_COMPLETION",
        "input": source_files, "binary": data_identity(binary), "source": data_identity(Path(__file__)),
        "preregistered_plan": "research/2026-10-05-gaussian-optimization-execution/TRACK_COMPLETION_CANDIDATE.md",
        "maximum_native_seconds": 600, "maximum_native_RSS_bytes": 32*1024**3,
        "input_camera_count": 1021, "fixed_poses": True, "fixed_intrinsics": True,
        "no_initial_frame_exclusion": True, "counted": False}
    atomic_json(args.output/"STATE.json", state)
    shutil.copyfile(Path(__file__), args.output/"source_snapshot.py")
    local_db = args.output/"database.db"
    shutil.copyfile(database, local_db)
    output_model = args.output/"model"
    output_model.mkdir()
    command = list(map(str, [binary, "point_triangulator", "--database_path", local_db,
        "--image_path", args.images.resolve(), "--input_path", model_path, "--output_path", output_model.resolve(),
        "--clear_points", "0", "--refine_intrinsics", "0", "--Mapper.fix_existing_frames", "1",
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0",
        "--Mapper.ba_use_gpu", "0", "--Mapper.num_threads", "8", "--Mapper.random_seed", "0",
        "--default_random_seed", "0", "--Mapper.tri_min_angle", "1.5",
        "--Mapper.tri_ignore_two_view_tracks", "1", "--Mapper.max_runtime_seconds", "590"]))
    state["command"] = command
    try:
        native_start = time.monotonic()
        peak = 0
        with (args.output/"triangulation.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="native_triangulation_running", native_pid=process.pid)
            atomic_json(args.output/"STATE.json", state)
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        peak = max(peak, sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True)))
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic()-native_start > 600 or peak > 32*1024**3:
                        raise RuntimeError("The native track-completion budget was exceeded")
                    time.sleep(.5)
            except BaseException:
                process.kill()
                process.wait()
                raise
        state["native_run"] = {"seconds": time.monotonic()-native_start, "peak_RSS_bytes": peak, "exit_code": process.returncode}
        if process.returncode:
            raise RuntimeError("Native triangulation failed; preserved log is authoritative")
        audit_command = [sys.executable, str(Path(__file__).with_name("audit_sfm_change.py")),
                         "--before", str(model_path), "--after", str(output_model), "--output", str(args.output/"common_audit")]
        with (args.output/"common_audit.log").open("w") as log:
            subprocess.run(audit_command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
        audit = json.loads((args.output/"common_audit/RESULT.json").read_text())
        state["common_observation_audit"] = audit
        if not audit["guard_passed"]:
            state.update(status="rejected_common_observation_or_fixed_camera_guard", metric_filter_not_run=True)
        else:
            state.update(status="metric_filter_running")
            atomic_json(args.output/"STATE.json", state)
            filtered = metric_filter(args, output_model)
            state["metric_filter"] = filtered
            state["status"] = "geometry_guards_passed_BA_pending" if filtered["passed"] else "rejected_original_geometry_support"
        unchanged = source_files == [data_identity(path) for path in sorted(model_path.glob("*.bin"))] + [data_identity(database)]
        state["source_unchanged"] = unchanged
        if not unchanged:
            raise ValueError("The source map or database changed during the diagnostic")
        state["elapsed_seconds"] = time.monotonic()-started
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({key: state[key] for key in ("status", "native_run", "source_unchanged", "elapsed_seconds")}), flush=True)
    except BaseException as error:
        state.update(status="failed_execution", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise


if __name__ == "__main__":
    main()
