"""Select useful views from one bounded pool, then register and refine them once."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from common import (MCAP, PACKAGE, Heartbeat, ba_config, ba_options, configure_stdout,
                    image_quality, load_mcap_tools, make_priors, read_json, solver_receipt, write_json)
from feature_localization import FeatureBank, estimate_pose, pose_curve
from run_prior_ba import observation_metrics, quantiles


def roi_contract(run):
    source = run.parents[1] / "04_outputs/box2_3dgs_display_v1_20260930/DISPLAY_EXPORT_RECEIPT.json"
    receipt = read_json(source)
    lo, hi = np.asarray(receipt["roi_minimum"]), np.asarray(receipt["roi_maximum"])
    return {"minimum": lo, "maximum": hi, "center": (lo + hi) / 2,
            "corners": np.asarray(list(product(*zip(lo, hi)))), "source": source,
            "boundary": "existing_development_ROI_for_two_stage_selection_not_independent_LiDAR"}


def target_window(camera, rough, roi):
    rotation = np.asarray(rough["R_WC"])
    center = np.asarray(rough["C_W_m"])
    xyz = (roi["corners"] - center) @ rotation
    focus = (roi["center"] - center) @ rotation
    if focus[2] <= 0 or np.any(xyz[:, 2] <= 0):
        return None
    pixels = camera.img_from_cam(xyz)
    focal_pixel = camera.img_from_cam(focus)
    if not np.isfinite(pixels).all() or not (0 <= focal_pixel[0] < camera.width and 0 <= focal_pixel[1] < camera.height):
        return None
    lo = np.floor(pixels.min(0)).astype(int)
    hi = np.ceil(pixels.max(0)).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [camera.width, camera.height])
    if np.any(hi - lo < 32):
        return None
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def view_features(rough, roi):
    vector = np.asarray(rough["C_W_m"]) - roi["center"]
    distance = float(np.linalg.norm(vector))
    azimuth = float(np.rad2deg(np.arctan2(vector[1], vector[0])) % 360)
    elevation = float(np.rad2deg(np.arctan2(vector[2], np.linalg.norm(vector[:2]))))
    key = f"{int(azimuth // 10):02d}_{int(np.floor(elevation / 15)):02d}"
    return {"view_bin": key, "azimuth_deg": azimuth, "elevation_deg": elevation,
            "target_distance_m": distance, "camera_height_m": rough["C_W_m"][2],
            "unit_target_view": vector / distance}


def augmented_prior_row(model, row, image_id):
    image = model.images[image_id]
    return {**row, "image_id": image.image_id, "camera_id": image.camera_id,
            "frame_id": image.frame_id, "frame_uid": f"augmentation:{image.name}",
            "corr_data_id": {"sensor_type": image.data_id.sensor_id.type.name,
                             "sensor_id": image.data_id.sensor_id.id, "data_id": image.data_id.id},
            "correlation_group": f"{row['left_pose_timestamp_ns']}_{row['right_pose_timestamp_ns']}",
            "quality_sharpness": row["global_quality"]["sharpness"],
            "quality_clipping": row["global_quality"]["clipped_fraction"],
            "position_covariance_m2": (np.eye(3) * .15**2).tolist(), "prior_active": True,
            "source_role": "recorded_current_mcap_odometry_augmentation", "producer": "unconfirmed",
            "quality_flags": ["development_quality_and_PnP_registration_passed"],
            "uncertainty_source": "predeclared_development_sigma015_not_measured_accuracy"}


def choose(run):
    output = run / "03_coverage_augmented"
    output.mkdir(exist_ok=False)
    heartbeat = Heartbeat(output / "STATE.json")
    start = time.monotonic()
    try:
        split = read_json(run / "00_contract/split_manifest.json")
        roi = roi_contract(run)
        curve = pose_curve()
        selected_case = read_json(run / "02_same_180_ba/SELECTION_RECEIPT.json")["selected_case"]
        model = pycolmap.Reconstruction(run / "02_same_180_ba" / selected_case / "model")
        camera = next(iter(model.cameras.values()))
        core, core_quality = [], []
        for row in split["core_train"]:
            rough = curve.at(row["camera_header_ns"])
            window = target_window(camera, rough, roi)
            image = cv2.imread(str(run / "00_contract/images" / row["image_name"]))
            quality = None if window is None else image_quality(image[window[1]:window[3], window[0]:window[2]])
            if quality:
                core_quality.append(quality["sharpness"])
            core.append({**row, **rough, **view_features(rough, roi), "target_window": window, "target_quality": quality})
        if not core_quality:
            raise RuntimeError("No core views support a target image window")
        target_threshold = float(np.quantile(core_quality, .05))
        McapIndex, _, decode, demosaic, _ = load_mcap_tools()
        index = McapIndex(MCAP)
        count = index.message_count("/sensing/camera/image_raw")
        pool = np.unique(np.linspace(round((count - 1) * .1), round((count - 1) * .68), 960, dtype=np.int64))
        excluded = {int(r["camera_frame"]) for r in split["core_train"] + split["extra_frames"]}
        frames = [int(f) for f in pool if int(f) not in excluded]
        write_json(output / "RESOURCE_PREFLIGHT.json", {"phase": "P3", "candidate_decode_count": len(frames),
            "maximum_additions_resource_cap": 64, "maximum_matching_pairs": 64 * 12,
            "maximum_wall_seconds": 7200, "maximum_BA_seconds": 1800,
            "quality_gate": split["quality_gate"], "target_sharpness_min": target_threshold,
            "target_threshold_source": "core_target_rectangle_sharpness_P05_development_heuristic",
            "target_clipped_fraction_max": .15, "view_bin_degrees": [10, 15],
            "desired_views_per_bin": 3, "novelty_angle_deg": 2., "novelty_translation_m": .05,
            "quality_upgrade_ratio": 1.25, "bridge_gap_frames": 100,
            "minimum_new_triangulation_angle_deg": 1.5, "maximum_new_point_reprojection_px": 2.,
            "ROI": roi, "H_D_cal_excluded": True,
            "boundary": "bounded_development_selection_resource_cap_is_not_a_frame_reduction_objective"})
        images = output / "candidate_images"
        images.mkdir()
        decisions, eligible = [], []
        heartbeat.phase = "bounded_candidate_decode_and_target_quality"
        for frame, message in index.iter_selected_messages("/sensing/camera/image_raw", frames):
            if time.monotonic() - start > 7200:
                raise TimeoutError("P3 selection reached its registered resource cap")
            timestamp, _, raw = decode(message["payload"])
            name = f"frame_{frame:05d}_{int(timestamp)}.jpg"
            row = {"camera_frame": int(frame), "camera_header_ns": int(timestamp), "image_name": name}
            try:
                rough = curve.at(int(timestamp))
            except ValueError as error:
                decisions.append({**row, "status": "rejected_time_support", "reason": str(error)})
                continue
            image = demosaic(raw)
            quality = image_quality(image)
            window = target_window(camera, rough, roi)
            target = None if window is None else image_quality(image[window[1]:window[3], window[0]:window[2]])
            good_global = quality["sharpness"] >= split["quality_gate"]["sharpness_min"] and quality["clipped_fraction"] <= .15
            good_target = target is not None and target["sharpness"] >= target_threshold and target["clipped_fraction"] <= .15
            row.update(rough, **view_features(rough, roi), global_quality=quality,
                       target_quality=target, target_window=window,
                       status="quality_eligible" if good_global and good_target else "rejected_quality_or_target_visibility")
            decisions.append(row)
            if good_global and good_target:
                ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 97])
                if not ok:
                    raise RuntimeError("Candidate image encoding failed")
                data.tofile(images / name)
                eligible.append(row)
        counts = Counter(r["view_bin"] for r in core if r["target_window"] is not None)
        best_quality = defaultdict(float)
        for row in core:
            if row["target_quality"]:
                best_quality[row["view_bin"]] = max(best_quality[row["view_bin"]], row["target_quality"]["sharpness"])
        core_times = sorted(int(r["camera_frame"]) for r in core)
        gaps = [(a, b) for a, b in zip(core_times, core_times[1:]) if b - a > 100]
        selected, consumed_bridges, existing = [], set(), list(core)
        for row in sorted(eligible, key=lambda x: x["target_quality"]["sharpness"], reverse=True):
            key = row["view_bin"]
            angles = np.rad2deg(np.arccos(np.clip(np.asarray([r["unit_target_view"] for r in existing]) @ row["unit_target_view"], -1, 1)))
            distances = np.linalg.norm(np.asarray([r["C_W_m"] for r in existing]) - row["C_W_m"], axis=1)
            novel = bool(np.all((angles >= 2.) | (distances >= .05)))
            bridge = next(((a, b) for a, b in gaps if (a, b) not in consumed_bridges
                           and a + .25 * (b - a) <= row["camera_frame"] <= a + .75 * (b - a)), None)
            reasons = []
            if counts[key] < 3:
                reasons.append("under_supported_view_bin")
            if row["target_quality"]["sharpness"] >= 1.25 * max(best_quality[key], target_threshold):
                reasons.append("target_quality_upgrade")
            if bridge:
                reasons.append("temporal_bridge")
            if novel and reasons and len(selected) < 64:
                row.update(status="selected_registration_pending", selection_reasons=reasons,
                           nearest_view_angle_deg=float(angles.min()), nearest_center_distance_m=float(distances.min()))
                selected.append(row)
                existing.append(row)
                counts[key] += 1
                best_quality[key] = max(best_quality[key], row["target_quality"]["sharpness"])
                if bridge:
                    consumed_bridges.add(bridge)
            else:
                row.update(status="not_selected", selection_reasons=reasons,
                           reason="resource_cap" if len(selected) >= 64 else "insufficient_new_view_or_quality_information")
        write_json(output / "SELECTION_MANIFEST.json", {"status": "frozen_before_registration", "selected_case": selected_case,
            "core_image_names": [r["image_name"] for r in core], "selected_additions": selected,
            "selected_count": len(selected), "eligible_count": len(eligible), "candidate_pool_count": len(frames),
            "ROI": roi, "target_threshold": target_threshold, "core_view_bin_counts": Counter(r["view_bin"] for r in core),
            "selected_view_bin_counts": counts, "selection_elapsed_seconds": time.monotonic() - start,
            "H_D_cal_overlap": 0})
        with (output / "frame_decisions.jsonl").open("x", encoding="utf-8") as stream:
            from common import serializable
            for row in decisions:
                stream.write(json.dumps(serializable(row), ensure_ascii=False) + "\n")
        print(f"P3 selected {len(selected)} additions from {len(frames)} candidates ({len(eligible)} quality eligible)")
    finally:
        heartbeat.close()


def triangulate_new_observations(model, image_id, keypoints, pairs, pixels):
    camera = next(iter(model.cameras.values()))
    image = model.images[image_id]
    created = 0
    by_neighbor = defaultdict(list)
    for neighbor, query, train, distance in sorted(pairs, key=lambda row: row[3]):
        by_neighbor[neighbor].append((query, train))
    for neighbor, candidates in by_neighbor.items():
        reference = model.images[neighbor]
        used_query, used_train, usable = set(), set(), []
        for query, train in candidates:
            if query in used_query or train in used_train:
                continue
            if image.points2D[query].has_point3D() or reference.points2D[train].has_point3D():
                continue
            used_query.add(query)
            used_train.add(train)
            usable.append((query, train))
        if not usable:
            continue
        xy1 = np.asarray([keypoints[q, :2] for q, _ in usable])
        xy2 = np.asarray([reference.points2D[t].xy for _, t in usable])
        normalized1 = camera.cam_from_img(xy1)
        normalized2 = camera.cam_from_img(xy2)
        pose1, pose2 = image.cam_from_world(), reference.cam_from_world()
        homogeneous = cv2.triangulatePoints(pose1.matrix(), pose2.matrix(), normalized1.T, normalized2.T).T
        valid_w = np.abs(homogeneous[:, 3]) > 1e-12
        xyz = np.zeros((len(homogeneous), 3))
        xyz[valid_w] = homogeneous[valid_w, :3] / homogeneous[valid_w, 3:4]
        camera1 = xyz @ pose1.rotation.matrix().T + pose1.translation
        camera2 = xyz @ pose2.rotation.matrix().T + pose2.translation
        projected1, projected2 = camera.img_from_cam(camera1), camera.img_from_cam(camera2)
        error1 = np.linalg.norm(projected1 - xy1, axis=1)
        error2 = np.linalg.norm(projected2 - xy2, axis=1)
        ray1, ray2 = xyz - image.projection_center(), xyz - reference.projection_center()
        norm_product = np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1)
        cosine = np.sum(ray1 * ray2, axis=1) / np.maximum(norm_product, 1e-12)
        angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        good = valid_w & np.isfinite(xyz).all(1) & (camera1[:, 2] > 0) & (camera2[:, 2] > 0)
        good &= (error1 <= 2.) & (error2 <= 2.) & (angles >= 1.5)
        for (query, train), point, accepted in zip(usable, xyz, good):
            if not accepted:
                continue
            px, py = np.rint(keypoints[query, :2]).astype(int)
            color = pixels[np.clip(py, 0, pixels.shape[0] - 1), np.clip(px, 0, pixels.shape[1] - 1), ::-1].copy()
            model.add_point3D(point, pycolmap.Track([pycolmap.TrackElement(image_id, query),
                                                   pycolmap.TrackElement(neighbor, train)]), color)
            created += 1
    return created


def target_support(model, roi):
    points = [p for p in model.points3D.values() if np.all(p.xyz >= roi["minimum"]) and np.all(p.xyz <= roi["maximum"])]
    return {"ROI_points": len(points), "ROI_track_length": quantiles([p.track.length() for p in points]),
            "ROI_observations": sum(p.track.length() for p in points),
            "boundary": "sparse_support_inside_fixed_development_ROI_not_surface_completeness"}


def register(run):
    output = run / "03_coverage_augmented"
    selection = read_json(output / "SELECTION_MANIFEST.json")
    registered_dir = output / "registered_model"
    registered_dir.mkdir(exist_ok=False)
    heartbeat = Heartbeat(output / "REGISTRATION_STATE.json")
    start = time.monotonic()
    try:
        model = pycolmap.Reconstruction(run / "02_same_180_ba" / selection["selected_case"] / "model")
        original_count = model.num_reg_images()
        original_ids = set(model.images)
        intrinsics = {i: c.params.copy() for i, c in model.cameras.items()}
        bank = FeatureBank(run)
        sys.path.insert(0, str(PACKAGE / "code/validated_fresh_runtime"))
        from metric_admission import camera_geometry_metrics
        policy = read_json(PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json")
        before_support = target_support(model, selection["ROI"])
        images = output / "images"
        images.mkdir()
        for name in selection["core_image_names"]:
            shutil.copy2(run / "00_contract/images" / name, images / name)
        frame_records, graph, new_rows = [], [], []
        heartbeat.phase = "SIFT_geometric_matching_and_PnP_registration"
        for offset, row in enumerate(selection["selected_additions"], 1):
            if time.monotonic() - start > 7200:
                raise TimeoutError("P3 registration reached the frozen two-hour budget")
            image_id = max(original_ids) + offset
            path = output / "candidate_images" / row["image_name"]
            keypoints, descriptors = bank.extra(path)
            neighbors = bank.neighbors(model, row, row["camera_header_ns"])
            correspondences, edges, pairs = bank.match(model, keypoints, descriptors, neighbors)
            graph.extend({"new_image_id": image_id, "new_image_name": row["image_name"], **edge} for edge in edges)
            result, info, localization, _ = estimate_pose(model, keypoints, correspondences)
            record = {"image_id": image_id, "image_name": row["image_name"], "features": len(keypoints),
                      "neighbors": neighbors, "matching_pairs": len(neighbors), **info}
            if result is None:
                frame_records.append(record)
                continue
            supported_points = [pycolmap.Point2D(keypoints[r["query"], :2], r["point_id"])
                                for r, accepted in zip(localization, result["inlier_mask"]) if accepted]
            geometry_image = pycolmap.Image(name=row["image_name"], points2D=supported_points,
                                           camera_id=next(iter(model.cameras)), image_id=image_id)
            geometry = camera_geometry_metrics(model, geometry_image, policy["camera_geometry_admission"])
            if not geometry["passed"]:
                record.update(status="rejected_geometry", geometry=geometry)
                frame_records.append(record)
                continue
            image = pycolmap.Image(name=row["image_name"], keypoints=keypoints[:, :2].astype(float),
                                   camera_id=next(iter(model.cameras)), image_id=image_id)
            model.add_image_with_trivial_frame(image, result["cam_from_world"])
            for correspondence, accepted in zip(localization, result["inlier_mask"]):
                if accepted:
                    model.add_observation(correspondence["point_id"], pycolmap.TrackElement(image_id, correspondence["query"]))
            matches = output / "matches"
            matches.mkdir(exist_ok=True)
            np.savez_compressed(matches / f"{image_id}.npz", pairs=np.asarray(pairs))
            pixels = cv2.imread(str(path))
            created = triangulate_new_observations(model, image_id, keypoints, pairs, pixels)
            shutil.copy2(path, images / path.name)
            record.update(status="registered", geometry=geometry, newly_triangulated_points=created,
                          rough_to_PnP_center_delta_m=float(np.linalg.norm(model.images[image_id].projection_center() - row["C_W_m"])))
            frame_records.append(record)
            new_rows.append(augmented_prior_row(model, row, image_id))
            print(f"Registered {row['image_name']}: {info['inliers']} inliers, {created} new points", flush=True)
        model.write(registered_dir)
        write_json(output / "view_graph.json", {"edges": graph, "attempted_matching_pairs": len(graph),
            "geometric_verified_edges": sum(e["geometric_inliers"] >= 15 for e in graph),
            "new_registered_views_connected_to_core": all(any(e["new_image_id"] == r["image_id"] and e["geometric_inliers"] >= 15 for e in graph)
                                                         for r in frame_records if r["status"] == "registered")})
        write_json(output / "REGISTRATION_RECEIPT.json", {"status": "completed", "selected_count": len(frame_records),
            "registered_additions": len(new_rows), "core_images": original_count, "total_registered": model.num_reg_images(),
            "frame_records": frame_records, "elapsed_seconds": time.monotonic() - start,
            "before_ROI_support": before_support, "after_registration_ROI_support": target_support(model, selection["ROI"]),
            "new_points": model.num_points3D() - 43641, "H_D_cal_used": False})
        refine_augmented(run, output, model, new_rows, intrinsics, heartbeat, start, bank)
    except Exception as error:
        write_json(output / "REGISTRATION_FAILURE.json", {"status": "failed_verification", "error": repr(error)})
        raise
    finally:
        heartbeat.close()


def refine_augmented(run, output, model, new_rows, intrinsics, heartbeat, start, bank, attempt=1):
    stage = output if attempt == 1 else output / f"refinement_attempt{attempt}"
    if attempt != 1:
        stage.mkdir(exist_ok=False)
    protocol = read_json(run / "00_contract/EVAL_PROTOCOL.json")
    rows = read_json(run / "01_priors/camera_prior_manifest.json")["images"] + new_rows
    groups = defaultdict(list)
    for row in rows:
        row["prior_active"] = False
        groups[(row["left_pose_index"], row["right_pose_index"])].append(row)
    for group in groups.values():
        max(group, key=lambda r: (r["quality_sharpness"], -r["image_id"]))["prior_active"] = True
    priors = make_priors(model, rows, .15)
    write_json(stage / "camera_prior_manifest.json", {"images": rows, "active_prior_count": len(priors),
        "covariance": "predeclared_sigma015_squared_identity_m2", "H_D_cal_overlap": 0})
    source = sqlite3.connect((run / "00_contract/database.db").as_uri() + "?mode=ro", uri=True)
    destination = sqlite3.connect(stage / "database.db")
    source.backup(destination)
    source.close()
    destination.execute("DELETE FROM pose_priors")
    destination.commit()
    destination.close()
    with pycolmap.Database.open(stage / "database.db") as database:
        for row in new_rows:
            image_id = row["image_id"]
            database.write_frame(model.frames[image_id], use_frame_id=True)
            actual = database.write_image(model.images[image_id], use_image_id=True)
            if actual != image_id:
                raise ValueError("New database and reconstruction image IDs disagree")
            keypoints, descriptors = bank.extra(output / "candidate_images" / row["image_name"])
            database.write_keypoints(image_id, keypoints)
            encoded = np.clip(np.rint(descriptors * 512), 0, 255).astype(np.uint8)
            database.write_descriptors(image_id, pycolmap.FeatureDescriptors(pycolmap.FeatureExtractorType.SIFT, encoded))
            with np.load(output / "matches" / f"{image_id}.npz") as stored:
                pairs = stored["pairs"]
                for neighbor in np.unique(pairs[:, 0]).astype(int):
                    matches = pairs[pairs[:, 0] == neighbor, 1:3].astype(np.uint32)
                    database.write_matches(image_id, neighbor, matches)
        for prior in priors:
            actual = database.write_pose_prior(prior, use_pose_prior_id=True)
            checked = database.read_pose_prior(actual, False)
            if not np.array_equal(checked.position, prior.position) or not np.array_equal(checked.position_covariance, prior.position_covariance):
                raise ValueError("Augmented prior database readback changed values")
        if database.num_pose_priors() != len(priors):
            raise ValueError("Augmented database prior count mismatch")
    write_json(output / "PRIOR_WRITE_READBACK.json", {"status": "passed", "prior_count": len(priors),
        "new_image_count": len(new_rows), "IDs_positions_covariances_verified": True,
        "database_path": stage / "database.db", "prior_manifest_path": stage / "camera_prior_manifest.json",
        "matching_database_mode": "geometrically_verified_new_pairs_PnP_registration_not_fresh_mapper"})
    before = observation_metrics(model)
    ba_start = time.monotonic()
    heartbeat.phase = "augmented_points_only_warmup"
    warmup = solver_receipt(pycolmap.create_default_ceres_bundle_adjuster(
        ba_options(protocol["max_warmup_iterations"]), ba_config(model, fixed_poses=True), model).solve())
    if not warmup["usable"]:
        raise RuntimeError("Augmented points-only warmup is unusable")
    config = ba_config(model)
    expected_visual = int(config.num_residuals(model))
    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    prior_options.prior_position_fallback_stddev = .15
    prior_options.alignment_ransac.max_error = .6
    prior_options.alignment_ransac.min_inlier_ratio = .2
    prior_options.alignment_ransac.random_seed = 20260809
    prior_options.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.CAUCHY
    prior_options.ceres.prior_position_loss_scale = protocol["prior_loss_scale"]
    heartbeat.phase = "augmented_position_prior_joint_BA"
    joint_options = ba_options(protocol["max_joint_iterations"])
    remaining = min(protocol["max_ba_wall_seconds"] - (time.monotonic() - ba_start),
                    protocol["max_augmented_sfm_wall_seconds"] - (time.monotonic() - start))
    if remaining <= 0:
        raise TimeoutError("The augmented SfM or combined BA wall-clock budget is exhausted")
    joint_options.ceres.solver_options.max_solver_time_in_seconds = remaining
    joint = solver_receipt(pycolmap.create_pose_prior_ceres_bundle_adjuster(
        joint_options, prior_options, config, priors, model).solve())
    if not joint["usable"] or joint["num_residuals"] != expected_visual + 3 * len(priors):
        raise RuntimeError("Augmented joint BA solution or active prior count is invalid")
    if joint["num_residual_blocks"] != expected_visual // 2 + len(priors):
        raise RuntimeError("Augmented joint BA residual block count is invalid")
    if any(not np.array_equal(c.params, intrinsics[i]) for i, c in model.cameras.items()):
        raise RuntimeError("Augmented BA changed the frozen intrinsics")
    model.update_point_3d_errors()
    final = output / "model"
    final.mkdir()
    model.write(final)
    after = observation_metrics(model)
    roi = read_json(output / "SELECTION_MANIFEST.json")["ROI"]
    write_json(output / "coverage_report.json", {"status": "completed_contract_verified",
        "core_images": 180, "registered_additions": len(new_rows), "registered_images": model.num_reg_images(),
        "registered_image_names": [im.name for im in model.images.values() if im.has_pose],
        "before_augmented_BA": before, "after_augmented_BA": after, "warmup": warmup, "joint": joint,
        "active_position_priors": len(priors), "prior_activation_verified": True,
        "post_BA_ROI_support": target_support(model, roi), "elapsed_registration_and_BA_seconds": time.monotonic() - start,
        "system_quality_gain": "not_yet_evaluated", "geometry_accuracy": "not_evaluated_missing_independent_reference",
        "top_surface_boundary": "view_selection_does_not_create_unobserved_surfaces"})
    print(f"Augmented BA complete: {model.num_reg_images()} cameras, {model.num_points3D()} points, {len(priors)} active priors")


def resume_refinement(run, attempt):
    output = run / "03_coverage_augmented"
    if (output / "coverage_report.json").exists() or (output / "model").exists():
        raise FileExistsError("Refinement already started; do not duplicate or overwrite it")
    if attempt == 1 and ((output / "database.db").exists() or (output / "camera_prior_manifest.json").exists()):
        raise FileExistsError("The partial first refinement is preserved; choose an explicit repair attempt")
    if attempt != 1 and read_json(run / "diagnostics/AUGMENTED_CONTRACT_TESTS.json")["status"] != "passed":
        raise RuntimeError("The augmented write-order regression preflight did not pass")
    registration = read_json(output / "REGISTRATION_RECEIPT.json")
    selection = read_json(output / "SELECTION_MANIFEST.json")
    model = pycolmap.Reconstruction(output / "registered_model")
    if model.num_reg_images() != registration["total_registered"]:
        raise ValueError("Preserved registration model count disagrees with its receipt")
    selected_by_name = {r["image_name"]: r for r in selection["selected_additions"]}
    new_rows = []
    for record in registration["frame_records"]:
        if record["status"] != "registered":
            continue
        row = selected_by_name[record["image_name"]]
        image = model.images[record["image_id"]]
        new_rows.append(augmented_prior_row(model, row, image.image_id))
    repair_label = "" if attempt == 1 else f"_attempt{attempt}"
    write_json(output / f"REFINEMENT_REPAIR_RECEIPT{repair_label}.json", {"status": "repair_started_from_verified_registration",
        "preserved_failure": output / "REGISTRATION_FAILURE.json",
        "root_causes": ["quality_sharpness_field_mismatch", "database_requires_frame_and_data_ID_before_image"],
        "fixes": ["use_quality_sharpness_consistently", "write_frame_before_write_image"],
        "registrations_reused": len(new_rows), "PnP_not_repeated": True, "BA_not_previously_started": True})
    heartbeat = Heartbeat(output / f"REFINEMENT_STATE{repair_label}.json")
    try:
        refine_augmented(run, output, model, new_rows, {i: c.params.copy() for i, c in model.cameras.items()},
                         heartbeat, time.monotonic() - registration["elapsed_seconds"], FeatureBank(run), attempt)
    except Exception as error:
        write_json(output / f"REFINEMENT_FAILURE{repair_label}.json", {"status": "failed_verification", "error": repr(error)})
        raise
    finally:
        heartbeat.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", choices=["select", "register", "refine"], required=True)
    parser.add_argument("--refine-attempt", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "select":
        choose(args.run.resolve())
    elif args.phase == "register":
        register(args.run.resolve())
    else:
        resume_refinement(args.run.resolve(), args.refine_attempt)


if __name__ == "__main__":
    configure_stdout()
    main()
