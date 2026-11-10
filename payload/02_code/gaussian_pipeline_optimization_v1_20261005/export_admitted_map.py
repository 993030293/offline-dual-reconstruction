"""Export an explicit training subset while preserving the full new-map record."""

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import psutil
import pycolmap

from run_metric_refinement import projection_validity
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identities(path):
    return [data_identity(p) for p in sorted(Path(path).glob("*.bin"))]


def filter_points(model, policy, lo, hi):
    removed = []
    model.update_point_3d_errors()
    for point_id in list(model.points3D):
        point = model.points3D[point_id]
        reasons = []
        if not np.isfinite(point.xyz).all() or np.any(point.xyz < lo) or np.any(point.xyz > hi):
            reasons.append("invalid_or_outside_metric_extent")
        if not np.isfinite(point.error) or point.error > policy["maximum_point_reprojection_error_px"]:
            reasons.append("reprojection_error")
        if point.track.length() < policy["minimum_point_track_length"]:
            reasons.append("track_length")
        if reasons:
            removed.append({"point3D_id": int(point_id), "reasons": reasons,
                "observations": [[int(e.image_id), int(e.point2D_idx)] for e in point.track.elements]})
            model.delete_point3D(point_id)
    return removed


def write_initial_ply(model, path):
    dtype = [(name, "<f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
    dtype += [(name, "u1") for name in ("red", "green", "blue")]
    vertices = np.zeros(model.num_points3D(), dtype=dtype)
    points = [model.points3D[i] for i in sorted(model.points3D)]
    xyz, rgb = np.asarray([p.xyz for p in points]), np.asarray([p.color for p in points])
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, index]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}"]
    header += ["property float "+name for name in ("x", "y", "z", "nx", "ny", "nz")]
    header += ["property uchar "+name for name in ("red", "green", "blue")]
    with path.open("xb") as stream:
        stream.write(("\n".join(header)+"\nend_header\n").encode("ascii"))
        stream.write(vertices.tobytes())
    recovered = pycolmap.Reconstruction()
    recovered.import_PLY(path)
    actual = list(recovered.points3D.values())
    expected_rows = np.column_stack((xyz.astype(np.float32), rgb))
    actual_rows = np.asarray([np.r_[p.xyz, p.color] for p in actual])
    def canonical(rows):
        return rows[np.lexsort(tuple(rows[:, index] for index in range(rows.shape[1]-1, -1, -1)))]
    # Reconstruction uses an unordered point map; compare the complete multiset.
    if len(actual) != len(points) or not np.array_equal(canonical(actual_rows), canonical(expected_rows)):
        raise ValueError("Official sparse PLY reader changed coordinates or colors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ba", "roles", "images", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"status": "validating_inputs", "source": data_identity(Path(__file__)),
        "maximum_seconds": 900, "maximum_RSS_bytes": 32*1024**3,
        "maximum_output_bytes": 30*1024**3, "counted": False}
    atomic_json(args.output/"STATE.json", state)
    try:
        if shutil.disk_usage(args.output).free < 20*1024**3:
            raise OSError("Insufficient disk headroom")
        ba, roles = read(args.ba/"RESULT.json"), read(args.roles)
        if (ba["status"] != "metric_interface_verified" or not ba["development_guard_passed"]
                or ba["selected"] != "weak_position_prior" or ba["common_observation_p95_delta_px"] > .05
                or roles["phase"] != "after_BA" or not roles["projection"]["valid"]):
            raise ValueError("New-map export requires the original guarded metric BA and post-BA roles")
        model_path = Path(ba["results"]["weak_position_prior"]["model"]).resolve()
        source = identities(model_path)
        role_ids = {Path(r["path"]).name: r["sha256"] for r in roles["source_identity"]}
        if {Path(r["path"]).name: r["sha256"] for r in source} != role_ids:
            raise ValueError("Post-BA role map identity differs from selected BA model")
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root/"02_code/box2_lio_guided_sfm_v1_20261002"))
        common = importlib.import_module("common")
        sys.path.insert(0, str(common.PACKAGE/"code/validated_fresh_runtime"))
        from metric_admission import camera_geometry_metrics
        policy_path = common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"
        if data_identity(policy_path)["sha256"] != roles["policy_source"]["sha256"]:
            raise ValueError("The geometry policy changed after role audit")
        policy = read(policy_path)
        model = pycolmap.Reconstruction(model_path)
        registered = {i: im.name for i, im in model.images.items() if im.has_pose}
        if set(registered.values()) != {r["image_name"] for r in roles["rows"]}:
            raise ValueError("Registered role denominator differs")
        full_copy = args.output/"full_opencv_model"
        full_copy.mkdir()
        model.write(full_copy)
        full_points = model.num_points3D()
        full_observations = model.compute_num_observations()
        centers = np.asarray([im.projection_center() for im in model.images.values() if im.has_pose])
        pp = policy["point_filter"]
        if not np.isfinite(centers).all() or np.linalg.norm(centers, axis=1).max() > pp["maximum_camera_radius_m"]:
            raise ValueError("The metric camera extent is invalid")
        lo, hi = centers.min(0)-pp["camera_aabb_margin_m"], centers.max(0)+pp["camera_aabb_margin_m"]
        removed_before = filter_points(model, pp, lo, hi)
        support_before = {im.name: camera_geometry_metrics(model, im, policy["camera_geometry_admission"])
                          for im in model.images.values() if im.has_pose}
        excluded = {name for name, row in support_before.items() if not row["passed"]}
        keep = set(registered.values())-excluded
        frames = {im.frame_id for im in model.images.values() if im.has_pose and im.name in excluded}
        if any(im.has_pose and im.frame_id in frames and im.name in keep for im in model.images.values()):
            raise ValueError("A shared rig frame would drop an admitted camera")
        for frame_id in sorted(frames):
            model.deregister_frame(frame_id)
        removed_after = filter_points(model, pp, lo, hi)
        support_after = {im.name: camera_geometry_metrics(model, im, policy["camera_geometry_admission"])
                         for im in model.images.values() if im.has_pose}
        atomic_json(args.output/"POINT_FILTER.json", {"before_subset": removed_before,
            "after_subset": removed_after, "minimum_track_length": pp["minimum_point_track_length"],
            "fallback_track_length_used": False})
        rows = [{**row, "support_after_point_filter": support_before[row["image_name"]],
            "training_role": "admitted" if row["image_name"] in keep else "excluded_insufficient_support",
            "support_in_training_subset": support_after.get(row["image_name"])} for row in roles["rows"]]
        receipt = {"status": "subset_checks_pending", "full_registered_denominator": len(registered),
            "raw_candidate_count": roles["raw_candidate_count"], "initial_count": roles["initial_count"],
            "original_registered_count": roles["registered_initial_count"],
            "initial_not_registered": roles["initial_not_registered"],
            "added_registered_names": roles["added_registered_names"],
            "protected_development_exclusion_count": roles["protected_development_exclusion_count"],
            "training_count": len(keep), "excluded_names": sorted(excluded), "rows": rows,
            "source_model": str(model_path), "source_identity": source, "post_BA_roles": data_identity(args.roles),
            "BA_receipt": data_identity(args.ba/"RESULT.json"), "policy": data_identity(policy_path),
            "full_points": full_points, "full_observations": full_observations,
            "training_points": model.num_points3D(), "training_observations": model.compute_num_observations(),
            "old_all_frame_failures_remain_failed": True, "counted": False}
        atomic_json(args.output/"TRAINING_ADMISSION.json", receipt)
        if (set(support_after) != keep or not all(r["passed"] for r in support_after.values())
                or len(keep) < policy["camera_geometry_admission"]["minimum_retained_cameras"]
                or model.num_points3D() < pp["minimum_retained_sparse_points"]):
            raise ValueError("Actual subset lost required support; no recursive exclusion or threshold relaxation")
        validity = projection_validity(model)
        atomic_json(args.output/"PROJECTION_VALIDITY.json", validity)
        if not validity["valid"]:
            raise ValueError("Actual training subset has invalid observed projections")
        filtered = args.output/"training_opencv_model"
        filtered.mkdir()
        model.write(filtered)
        reread = pycolmap.Reconstruction(filtered)
        if {im.name for im in reread.images.values() if im.has_pose} != keep:
            raise ValueError("Persisted training map changed camera membership")
        for name in sorted(keep):
            if not (args.images/name).is_file():
                raise FileNotFoundError(args.images/name)
        state.update(status="undistorting", training_count=len(keep))
        atomic_json(args.output/"STATE.json", state)
        dataset = args.output/"dataset"
        pycolmap.undistort_images(dataset, filtered, args.images, image_names=sorted(keep),
            output_type="COLMAP", num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
        pinhole = pycolmap.Reconstruction(dataset/"sparse")
        if (any(c.model_name != "PINHOLE" for c in pinhole.cameras.values())
                or {im.name for im in pinhole.images.values() if im.has_pose} != keep
                or {p.name for p in (dataset/"images").iterdir() if p.is_file()} != keep
                or set(model.points3D) != set(pinhole.points3D)):
            raise ValueError("Undistortion changed camera/image/point membership or model type")
        pose_error, point_error = 0., 0.
        for image in pinhole.images.values():
            if not image.has_pose:
                continue
            pose_error = max(pose_error, float(np.max(np.abs(image.cam_from_world().matrix()
                - model.images[image.image_id].cam_from_world().matrix()))))
            with Image.open(dataset/"images"/image.name) as photo:
                camera = pinhole.cameras[image.camera_id]
                if photo.size != (camera.width, camera.height):
                    raise ValueError("Image size differs from full PINHOLE intrinsics")
        for point_id, point in pinhole.points3D.items():
            point_error = max(point_error, float(np.linalg.norm(point.xyz-model.points3D[point_id].xyz)))
        final_support = [camera_geometry_metrics(pinhole, im, policy["camera_geometry_admission"])
                         for im in pinhole.images.values() if im.has_pose]
        final_validity = projection_validity(pinhole)
        if max(pose_error, point_error) > 1e-9 or not all(r["passed"] for r in final_support) or not final_validity["valid"]:
            raise ValueError("Undistorted final training map failed geometry/coordinate preservation")
        sparse = dataset/"sparse/0"
        sparse.mkdir()
        pinhole.write(sparse)
        pinhole.write_text(sparse)
        write_initial_ply(pinhole, sparse/"points3D.ply")
        memory = psutil.Process().memory_info()
        peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss))
        output_bytes = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
        if peak > state["maximum_RSS_bytes"] or time.monotonic()-started > 900 or output_bytes > 30*1024**3:
            raise RuntimeError("Registered export resource budget exceeded")
        if source != identities(model_path):
            raise RuntimeError("The immutable BA source changed")
        receipt.update(status="export_and_admission_verified_training_pending", dataset=str(dataset.resolve()),
            training_opencv_model=str(filtered.resolve()), full_map_copy=str(full_copy.resolve()),
            final_pinhole_support=final_support, final_projection=final_validity,
            max_pose_element_delta=pose_error, max_point_delta_m=point_error,
            PINHOLE_cameras={str(i): common.serializable(c.todict()) for i, c in pinhole.cameras.items()},
            source_unchanged=True, elapsed_seconds=time.monotonic()-started,
            peak_RSS_bytes=peak, output_bytes=output_bytes, independent_accuracy_claim=False)
        atomic_json(args.output/"TRAINING_ADMISSION.json", receipt)
        state.update(status=receipt["status"], training_count=len(keep), excluded_names=sorted(excluded),
            elapsed_seconds=receipt["elapsed_seconds"], peak_RSS_bytes=peak)
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="failed_export_no_training_admission", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise


if __name__ == "__main__":
    main()
