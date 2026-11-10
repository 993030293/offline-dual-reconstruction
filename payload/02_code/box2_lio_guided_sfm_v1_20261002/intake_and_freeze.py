"""Freeze Box2 inputs, roles and evaluation policy before candidate optimization."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pycolmap

from common import (BASE, COLMAP, MCAP, PACKAGE, PLAN, PROJECT, VISUAL, Heartbeat,
                    configure_stdout, digest_data, image_quality, load_mcap_tools,
                    metric_initial_model, read_json, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    contract = run / "00_contract"
    contract.mkdir()
    heartbeat = Heartbeat(run / "EXECUTION_STATE.json")
    start = time.monotonic()
    try:
        original = read_json(VISUAL / "image_manifest.json")
        if len(original) != 180 or len({r["image_name"] for r in original}) != 180:
            raise ValueError("Unexpected baseline image identities")
        if MCAP.stat().st_size != 16_538_906_825:
            raise ValueError("Raw recording byte count changed")
        database = VISUAL / "database.db"
        if database.with_name("database.db-wal").exists() or database.with_name("database.db-journal").exists():
            raise RuntimeError("Baseline database has an active journal")
        heartbeat.phase = "copying_read_only_baseline"
        source = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        destination = sqlite3.connect(contract / "database.db")
        with destination:
            source.backup(destination)
        identities = dict(source.execute("SELECT name,image_id FROM images"))
        table_counts = {name: source.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                        for name in ("images", "keypoints", "descriptors", "matches", "two_view_geometries")}
        source.close()
        destination.close()
        if set(identities) != {row["image_name"] for row in original}:
            raise ValueError("Database and manifest image identities disagree")
        images = contract / "images"
        images.mkdir()
        for row in original:
            path = VISUAL / "images" / row["image_name"]
            if digest_data(path) != row["image_sha256"]:
                raise ValueError(f"Image identity mismatch: {path}")
            shutil.copy2(path, images / path.name)
        reconstruction = metric_initial_model()
        if reconstruction.num_reg_images() != 180 or reconstruction.num_points3D() != 43641:
            raise ValueError("Unexpected raw visual model counts")
        if any(camera.model_name != "OPENCV" for camera in reconstruction.cameras.values()):
            raise ValueError("The common BA initialization must retain OPENCV pixels")
        initial = contract / "initial_model"
        initial.mkdir()
        reconstruction.write(initial)
        data_ids = {}
        for path in [database, *sorted((VISUAL / "sparse/0").glob("*.bin"))]:
            data_ids[str(path)] = digest_data(path)
        write_json(contract / "BASELINE_MANIFEST.json", {
            "status": "frozen", "registered_images": 180, "points": 43641,
            "model_domain": "original_OPENCV_pixels_transformed_to_metric",
            "source_images": original, "database_table_counts": table_counts,
            "data_sha256": data_ids, "plan_path": PLAN, "plan_revision": "r1_20261002T005049_9bdfd943",
            "raw_path": MCAP, "raw_full_hash_recomputed": False, "active_git_worktree": False})
        heartbeat.phase = "reading_mcap_topics_and_covariance"
        McapIndex, Cdr, decode, demosaic, color = load_mcap_tools()
        index = McapIndex(MCAP)
        topics = [{"topic": c.topic, "count": index.message_count(c.topic),
                   "schema": index.schemas[c.schema_id], "encoding": c.encoding}
                  for c in index.channels.values()]
        covariances, poses, stamps, frames = [], [], [], set()
        for message in index.iter_messages("/localization/pose/odom"):
            cdr = Cdr(message["payload"])
            sec, ns, parent = cdr.header()
            child = cdr.string()
            poses.append([cdr.f64() for _ in range(7)])
            covariances.append([cdr.f64() for _ in range(36)])
            stamps.append(int(sec) * 1_000_000_000 + int(ns))
            frames.add((parent, child))
        cov = np.asarray(covariances).reshape(-1, 6, 6)
        tum = np.loadtxt(BASE / "03_metric_context/metric_odometry.tum")
        if len(stamps) != len(tum) or not np.array_equal(np.asarray(stamps), np.rint(tum[:, 0] * 1e9).astype(np.int64)):
            raise ValueError("Raw odometry timestamps disagree with the preserved trajectory")
        pose_delta = float(np.max(np.abs(np.asarray(poses) - tum[:, 1:])))
        if pose_delta > 1e-8 or frames != {("odom", "base_link")}:
            raise ValueError("Raw odometry pose/frame contract changed")
        covariance_status = "all_zero_unusable" if np.all(cov == 0) else "present_not_yet_calibrated"
        write_json(contract / "RAW_SOURCE_RECEIPT.json", {
            "topics": topics, "odometry_count": len(stamps), "frames": sorted(frames),
            "producer": "unconfirmed_no_publisher_algorithm_in_recorded_topic_schema",
            "covariance_status": covariance_status, "covariance_first_6x6": cov[0],
            "covariance_nonzero_samples": int(np.sum(np.any(cov != 0, axis=(1, 2)))),
            "maximum_raw_vs_tum_pose_delta": pose_delta, "color_contract": color,
            "raw_file_bytes": MCAP.stat().st_size, "raw_full_hash_recomputed": False})
        heartbeat.phase = "freezing_extra_evaluation_frames"
        camera_topic = "/sensing/camera/image_raw"
        total = index.message_count(camera_topic)
        first, last = round((total - 1) * .1), round((total - 1) * .68)
        pool = np.unique(np.linspace(first, last, 960, dtype=np.int64))
        core = {int(row["camera_frame"]) for row in original}
        available = np.asarray([int(x) for x in pool if int(x) not in core])
        chosen = available[np.linspace(0, len(available) - 1, 48, dtype=int)]
        role = {int(frame): ("H" if i % 2 == 0 else "D_cal") for i, frame in enumerate(chosen)}
        threshold = float(np.quantile([row["sharpness"] for row in original], .05))
        extra = []
        (contract / "extra_images").mkdir()
        for frame, message in index.iter_selected_messages(camera_topic, chosen.tolist()):
            stamp, _, raw = decode(message["payload"])
            image = demosaic(raw)
            quality = image_quality(image)
            name = f"frame_{frame:05d}_{int(stamp)}.jpg"
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 97])
            if not ok:
                raise RuntimeError("Camera image encoding failed")
            encoded.tofile(contract / "extra_images" / name)
            extra.append({"image_name": name, "camera_frame": int(frame),
                "camera_header_ns": int(stamp), "role": role[int(frame)],
                "quality_eligible": quality["sharpness"] >= threshold and quality["clipped_fraction"] <= .15,
                **quality})
        write_json(contract / "split_manifest.json", {
            "status": "frozen_before_candidates", "core_train": original, "extra_frames": extra,
            "selection": "48 uniformly spaced non-core candidates; alternating H and D_cal",
            "H_usage": "localization_and_evaluation_only_no_map_update_no_candidate_selection",
            "D_cal_usage": "development_calibration_only_excluded_from_training_and_augmentation",
            "quality_gate": {"sharpness_min": threshold, "clipped_fraction_max": .15,
                "boundary": "relative_baseline_heuristic_not_absolute_quality_certification"}})
        write_json(contract / "EVAL_PROTOCOL.json", {
            "status": "frozen_before_candidates", "development_recording": True,
            "core_image_count": 180, "common_observations": "all_original_visual_model_tracks_before_filter",
            "same_frame_cases": ["A1_visual", "B_w_sigma015", "B_s_sigma005"],
            "sigma_candidates_m": [.15, .05], "prior_loss": "CAUCHY", "prior_loss_scale": 2.5,
            "visual_loss": "HUBER", "visual_loss_scale": 1.0, "prior_group": "shared_odometry_bracket",
            "max_warmup_iterations": 20, "max_joint_iterations": 100,
            "max_ba_wall_seconds": 1800, "max_augmented_sfm_wall_seconds": 7200,
            "max_gaussian_wall_seconds": 1800, "gaussian_iterations": 10000, "gaussian_seed": 0,
            "common_reprojection_p95_development_tolerance_px": .05,
            "tolerance_source": "predeclared_engineering_guard_of_one_twentieth_pixel_not_statistical_or_GT_precision",
            "H_metrics": "descriptive_development_leaveout_no_automatic_promotion",
            "H_loc_H_eval_split": "deterministic_disjoint_correspondences_at_final_evaluation",
            "absolute_pose_accuracy": "not_evaluated_missing_independent_GT",
            "reduction_required": False})
        write_json(contract / "INPUT_CONTRACT.json", {
            "status": "passed", "coordinate_convention": "T_A_B_maps_B_to_A",
            "body_pose": "odom_from_base_link", "extrinsic": "base_link_from_camera_optical",
            "BA_camera_domain": "OPENCV_original_feature_pixels",
            "downstream_camera_domain": "PINHOLE_new_undistortion_after_BA",
            "time_domain": "sensor_header_relative_epoch", "time_offset_ns": 0,
            "time_offset_precision": "unconfirmed", "max_bracket_span_ns": 120_000_000,
            "extrapolation": "forbidden", "offline_future_bracketing": True})
        write_json(contract / "RESOURCE_PREFLIGHT.json", {
            "elapsed_seconds": time.monotonic() - start, "pycolmap_version": pycolmap.__version__,
            "cpu_threads_per_ba": 8, "logical_cpu_count": psutil.cpu_count(),
            "available_memory_bytes": psutil.virtual_memory().available,
            "free_project_disk_bytes": shutil.disk_usage(PROJECT).free,
            "source_database_bytes": database.stat().st_size,
            "new_input_bytes": sum(p.stat().st_size for p in contract.rglob("*") if p.is_file()),
            "phase": "P0", "complete_system_runtime_forecast": "not_measured"})
        write_json(contract / "P0_RECEIPT.json", {"status": "passed", "elapsed_seconds": time.monotonic() - start,
            "core_images": 180, "extra_roles": {r: sum(x["role"] == r for x in extra) for r in ("H", "D_cal")},
            "covariance_status": covariance_status, "input_database_preserved": digest_data(database) == data_ids[str(database)]})
        print("P0 input contracts and roles frozen", flush=True)
    finally:
        heartbeat.close()


if __name__ == "__main__":
    configure_stdout()
    main()
