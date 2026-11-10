"""Localize additional images with the preserved SIFT identity and fixed intrinsics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from common import BASE, PoseCurve, read_json, write_json
from run_prior_ba import quantiles


def pose_curve():
    return PoseCurve(BASE / "03_metric_context/metric_odometry.tum",
                     BASE / "03_metric_context/T_base_link_camera_optical.json")


def feature_protocol(run):
    path = run / "00_contract/LOCALIZATION_PROTOCOL.json"
    if not path.exists():
        write_json(path, {"status": "frozen_before_extra_frame_localization",
            "SIFT_max_features": 12000, "SIFT_peak_threshold": .004,
            "SIFT_normalization": "L1_ROOT", "new_feature_device": "CPU",
            "existing_features": "original_GPU_database_unchanged",
            "ratio_max": .8, "normalized_distance_max": .7, "mutual_check": True,
            "fundamental_max_error_px": 4., "fundamental_min_inliers": 15,
            "PnP_max_error_px": 4., "PnP_min_inliers": 30,
            "max_temporal_neighbors": 4, "max_pose_neighbors": 8,
            "H_correspondence_split": "query_keypoint_index_even_for_loc_odd_for_eval_unique_3D_ids",
            "H_refinement": "pose_only_no_training_map_update",
            "target_ROI": "fixed_existing_development_ROI_not_independent_LiDAR_reference"})
    return read_json(path)


class FeatureBank:
    def __init__(self, run):
        self.run = run
        self.protocol = feature_protocol(run)
        cv2.setNumThreads(8)
        options = pycolmap.FeatureExtractionOptions()
        options.sift.max_num_features = self.protocol["SIFT_max_features"]
        options.sift.peak_threshold = self.protocol["SIFT_peak_threshold"]
        self.extractor = pycolmap.Sift(options, pycolmap.Device.cpu)
        self.original = {}
        connection = sqlite3.connect((run / "00_contract/database.db").as_uri() + "?mode=ro", uri=True)
        rows = connection.execute("SELECT images.image_id,images.name,keypoints.rows,keypoints.cols,keypoints.data,descriptors.rows,descriptors.cols,descriptors.data FROM images JOIN keypoints USING(image_id) JOIN descriptors USING(image_id)")
        for image_id, name, kr, kc, kd, dr, dc, dd in rows:
            keypoints = np.frombuffer(kd, np.float32).reshape(kr, kc).copy()
            descriptors = np.frombuffer(dd, np.uint8).reshape(dr, dc).astype(np.float32) / 512.
            if kr != dr or dc != 128:
                raise ValueError("Original SIFT database contract changed")
            self.original[int(image_id)] = (name, keypoints, descriptors)
        connection.close()
        self.cache = run / "feature_cache"
        self.cache.mkdir(exist_ok=True)

    def extra(self, path):
        target = self.cache / (path.stem + ".npz")
        if target.exists():
            with np.load(target) as stored:
                return stored["keypoints"].copy(), stored["descriptors"].copy()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Image decode failed: {path}")
        keypoints, descriptors = self.extractor.extract(gray)
        np.savez_compressed(target, keypoints=keypoints, descriptors=descriptors)
        return keypoints, descriptors

    def neighbors(self, model, rough, timestamp):
        available = [i for i in sorted(model.images) if i in self.original]
        center = np.asarray(rough["C_W_m"])
        forward = np.asarray(rough["R_WC"])[:, 2]
        temporal = sorted(available, key=lambda i: abs(int(self.original[i][0].split("_")[2].split(".")[0]) - timestamp))[:4]
        scores = {}
        for i in available:
            image = model.images[i]
            view = image.cam_from_world().rotation.matrix().T[:, 2]
            scores[i] = np.linalg.norm(image.projection_center() - center) + .5 * (1 - np.clip(view @ forward, -1, 1))
        spatial = sorted(available, key=lambda i: scores[i])[:8]
        return list(dict.fromkeys(temporal + spatial))

    def match(self, model, keypoints, descriptors, neighbors, geometric=True):
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        correspondences, edges, all_pairs = [], [], []
        for image_id in neighbors:
            name, train_keypoints, train_descriptors = self.original[image_id]
            image = model.images[image_id]
            if len(train_keypoints) != len(image.points2D):
                raise ValueError("Original keypoint and map observation identity mismatch")
            if not np.allclose(train_keypoints[:, :2], np.asarray([p.xy for p in image.points2D]), atol=1e-3):
                raise ValueError("Original SIFT pixels and map pixels disagree")
            nearest = matcher.knnMatch(descriptors, train_descriptors, k=2)
            reverse = matcher.match(train_descriptors, descriptors)
            reverse_map = {m.queryIdx: m.trainIdx for m in reverse}
            pairs = [m for pair in nearest if len(pair) == 2 for m, n in [pair]
                     if m.distance < .8 * n.distance and m.distance <= .7
                     and reverse_map.get(m.trainIdx) == m.queryIdx]
            raw_count = len(pairs)
            matrix = None
            if geometric and len(pairs) >= 15:
                xy1 = np.asarray([keypoints[m.queryIdx, :2] for m in pairs])
                xy2 = np.asarray([train_keypoints[m.trainIdx, :2] for m in pairs])
                cv2.setRNGSeed(20261002)
                matrix, mask = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, 4., .999, 10000)
                pairs = [] if matrix is None or mask is None else [m for m, ok in zip(pairs, mask.ravel()) if ok]
            if geometric and len(pairs) < 15:
                pairs = []
            edges.append({"train_image_id": image_id, "train_image_name": name,
                          "mutual_ratio_matches": raw_count, "geometric_inliers": len(pairs),
                          "F_from_query_localization_features": None if matrix is None else matrix.tolist()})
            for match in pairs:
                all_pairs.append((image_id, match.queryIdx, match.trainIdx, match.distance))
                point = image.points2D[match.trainIdx]
                if point.has_point3D():
                    correspondences.append((match.distance, match.queryIdx, int(point.point3D_id), image_id, match.trainIdx))
        unique, used_query, used_point = [], set(), set()
        for distance, query, point_id, image_id, train_index in sorted(correspondences):
            if query in used_query or point_id in used_point:
                continue
            used_query.add(query)
            used_point.add(point_id)
            unique.append({"query": query, "point_id": point_id, "train_image_id": image_id,
                           "train_index": train_index, "distance": float(distance)})
        return unique, edges, all_pairs


def estimate_pose(model, keypoints, correspondences, split=False):
    camera = next(iter(model.cameras.values()))
    localization = [r for r in correspondences if not split or r["query"] % 2 == 0]
    evaluation = [r for r in correspondences if split and r["query"] % 2 == 1]
    info = {"correspondences": len(correspondences), "loc_correspondences": len(localization),
            "eval_correspondences": len(evaluation), "status": "failed_insufficient_support"}
    if len(localization) < 30:
        return None, info, localization, evaluation
    xy = np.asarray([keypoints[r["query"], :2] for r in localization])
    xyz = np.asarray([model.points3D[r["point_id"]].xyz for r in localization])
    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = 4.
    options.ransac.min_inlier_ratio = .1
    options.ransac.random_seed = 20261002
    options.ransac.num_threads = 8
    refinement = pycolmap.AbsolutePoseRefinementOptions()
    refinement.refine_focal_length = refinement.refine_extra_params = False
    result = pycolmap.estimate_and_refine_absolute_pose(xy, xyz, camera, options, refinement)
    if result is None or result["num_inliers"] < 30:
        info.update(status="failed_PnP", inliers=0 if result is None else result["num_inliers"])
        return None, info, localization, evaluation
    pose = result["cam_from_world"]
    inliers = np.asarray(result["inlier_mask"], dtype=bool)
    projected = camera.img_from_cam(xyz @ pose.rotation.matrix().T + pose.translation)
    errors = np.linalg.norm(projected - xy, axis=1)
    info.update(status="localized", inliers=int(inliers.sum()),
                inlier_fraction=float(inliers.mean()), loc_inlier_reprojection_px=quantiles(errors[inliers]),
                camera_center_m=pose.inverse().translation.tolist(), R_CW=pose.rotation.matrix().tolist(),
                t_CW=pose.translation.tolist())
    if evaluation:
        eval_xy = np.asarray([keypoints[r["query"], :2] for r in evaluation])
        eval_xyz = np.asarray([model.points3D[r["point_id"]].xyz for r in evaluation])
        eval_projected = camera.img_from_cam(eval_xyz @ pose.rotation.matrix().T + pose.translation)
        info["eval_reprojection_px_all_unused_correspondences"] = quantiles(np.linalg.norm(eval_projected - eval_xy, axis=1))
        info["loc_eval_3D_overlap"] = len({r["point_id"] for r in localization} & {r["point_id"] for r in evaluation})
    return result, info, localization, evaluation
