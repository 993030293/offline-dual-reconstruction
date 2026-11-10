"""Resume verified current-campaign rows without changing pose estimation or matches."""

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
import localize_evaluation as entry
from atomic_json_windows import atomic_json, read_json
from runtime import data_identity


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, (float, np.floating)) or bool(np.isfinite(value))


def resume_rows(previous, side, evaluation, camera):
    contract = read_json(previous / "CONTRACT.json")
    if contract["source_identity"] != [data_identity(Path(row["path"])) for row in contract["source_identity"]]:
        raise ValueError("Current-campaign model, features or evaluation identity changed")
    expected = {"mode": "optimized", "PnP_threads": 1, "SIFT_and_OpenCV_CPU_threads": 4,
                "SIFT_max_features": 12000, "SIFT_peak_threshold": .004, "all_evaluation_count": 24}
    if any(contract[key] != value for key, value in expected.items()):
        raise ValueError("Completed rows have different native options")
    rows = read_json(previous / "PARTIAL_RESULTS.json")
    if len(rows) >= 24 or [row["image_name"] for row in rows] != [row["image_name"] for row in evaluation[:len(rows)]]:
        raise ValueError("Only a verified ordered current-campaign prefix can be resumed")
    for row in rows:
        if not finite_tree(row) or row["status"] != "localized" or row["inliers"] < 30 or row["loc_eval_3D_overlap"] != 0:
            raise ValueError("An incomplete or invalid pose cannot be reused")
        if not np.isfinite(np.asarray(row["R_CW"])).all() or not np.isfinite(np.asarray(row["t_CW"])).all():
            raise ValueError("A reused pose contains nonfinite values")
        raw = side / "initial/evaluation_images" / row["image_name"]
        if row["raw_image"] != data_identity(raw) or not Path(row["image_reference"]).is_file():
            raise ValueError("A completed reference image changed or is missing")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(entry.__file__).read_text(encoding="utf-8")
    anchors = {"results, requests = [], []": "results, requests = resume_current_rows(evaluation, camera), []",
               "for source in evaluation:": "for source in evaluation[len(results):]:"}
    for before, after in anchors.items():
        if source.count(before) != 1:
            raise ValueError("A localization recovery source anchor changed")
        source = source.replace(before, after)
    resumed = []
    diagnostics = []
    original_pose = entry.estimate_pose

    def checked_pose(model, keypoints, correspondences, split=False):
        solution, info, used, unused = original_pose(model, keypoints, correspondences, split)
        statistic = info.get("eval_reprojection_px_all_unused_correspondences")
        required = {key: value for key, value in info.items() if key != "eval_reprojection_px_all_unused_correspondences"}
        if not finite_tree(required):
            raise FloatingPointError("Pose or localization inlier diagnostics contain nonfinite values")
        if statistic is not None and not finite_tree(statistic):
            if solution is None:
                raise FloatingPointError("An undefined pose cannot be accepted")
            pose = solution["cam_from_world"]
            if not np.isfinite(pose.matrix()).all():
                raise FloatingPointError("The native pose is nonfinite")
            xyz = np.asarray([model.points3D[row["point_id"]].xyz for row in unused])
            cam_xyz = xyz @ pose.rotation.matrix().T + pose.translation
            projected = next(iter(model.cameras.values())).img_from_cam(cam_xyz)
            invalid = ~np.isfinite(projected).all(axis=1)
            if not invalid.any() or not np.isfinite(cam_xyz).all():
                raise FloatingPointError("An unexplained nonfinite diagnostic must not be hidden")
            record = {"status": "all_unused_statistic_undefined_nonprojectable_matches",
                "complete_unused_denominator": len(unused), "nonprojectable_count": int(invalid.sum()),
                "nonpositive_depth_count": int((cam_xyz[:,2] <= 0).sum()), "all_matches_retained": True,
                "finite_subset_is_not_reported_as_all_unused_error": True, "PnP_and_render_pose_unchanged": True}
            np.savez(args.output / ("unused_projection_"+str(len(diagnostics))+".npz"),
                camera_xyz=cam_xyz, projected_pixels=projected,
                point_ids=np.asarray([row["point_id"] for row in unused]),
                query_ids=np.asarray([row["query"] for row in unused]))
            diagnostics.append(record)
            info["eval_reprojection_px_all_unused_correspondences"] = None
            info["unused_projection_diagnostic"] = record
        return solution, info, used, unused

    def resume_current_rows(evaluation, camera):
        rows = resume_rows(args.previous, args.side, evaluation, camera)
        resumed.extend(row["image_name"] for row in rows)
        for row in evaluation[len(rows):]:
            name = Path(row["image_name"]).stem+".npz"
            cached = args.previous / "feature_cache" / name
            if cached.exists():
                shutil.copyfile(cached, args.output / "feature_cache" / name)
        return rows

    namespace = dict(vars(entry))
    namespace.update(resume_current_rows=resume_current_rows)
    exec(compile(source, entry.__file__, "exec"), namespace)
    namespace["estimate_pose"] = checked_pose
    if namespace["main"].__globals__["estimate_pose"] is not checked_pose:
        raise RuntimeError("The diagnostic guard is not bound to the executed entry")
    original_argv = sys.argv
    sys.argv = [entry.__file__, "--side", str(args.side), "--mode", "optimized", "--output", str(args.output)]
    try:
        namespace["main"]()
        result = read_json(args.output / "LOCALIZATION.json")
        if result["status"] != "all24_localized":
            raise ValueError("The original complete24 localization gate did not pass")
        atomic_json(args.output / "RECOVERY_RECEIPT.json", {"status": "all24_localized_current_campaign_prefix_resumed",
            "resumed_names": resumed, "resumed_count": len(resumed), "newly_completed_count": 24-len(resumed),
            "previous": str(args.previous), "source": data_identity(Path(__file__)),
            "native_entry": data_identity(Path(entry.__file__)), "undefined_optional_diagnostics": diagnostics,
            "map_pose_feature_and_threshold_changes": False, "failed_original_outputs_preserved": True})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
