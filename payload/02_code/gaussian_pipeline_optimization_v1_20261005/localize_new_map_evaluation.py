"""Freeze all development cameras in the new map without importing old poses."""

import argparse
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import cv2
import numpy as np
from PIL import Image
import psutil
import pycolmap

from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def transfer_regions(old_rows, old_pinhole, old_raw, new_raw, new_pinhole):
    """Transfer frozen rectangles through the shared distorted sensor pixels."""
    def project(camera, xy):
        return camera.img_from_cam(np.column_stack((xy, np.ones(len(xy)))))
    def forward(xy):
        raw_pixels = project(old_raw, old_pinhole.cam_from_img(xy))
        return project(new_pinhole, new_raw.cam_from_img(raw_pixels))
    def backward(xy):
        raw_pixels = project(new_raw, new_pinhole.cam_from_img(xy))
        return project(old_pinhole, old_raw.cam_from_img(raw_pixels))
    regions, checks = {}, []
    for row in old_rows:
        x1, y1, x2, y2 = row["target_window_PINHOLE"]
        xs, ys = np.linspace(x1, x2, 65), np.linspace(y1, y2, 65)
        edge = np.vstack((np.column_stack((xs, np.full(65, y1))), np.column_stack((xs, np.full(65, y2))),
                          np.column_stack((np.full(65, x1), ys)), np.column_stack((np.full(65, x2), ys))))
        mapped = forward(edge)
        inverse_error = float(np.max(np.abs(backward(mapped)-edge)))
        if not np.isfinite(mapped).all() or inverse_error > 1e-6:
            raise ValueError("Calibration-only image-region transfer failed its inverse oracle")
        window = np.rint(np.r_[mapped.min(0), mapped.max(0)]).astype(int).tolist()
        if window[0] < 0 or window[1] < 0 or window[2] > new_pinhole.width or window[3] > new_pinhole.height:
            raise ValueError("Transferred frozen rectangle lies outside the new reference image")
        regions[row["image_name"]] = window
        checks.append({"image_name": row["image_name"], "old_window": row["target_window_PINHOLE"],
            "new_window": window, "max_edge_movement_px": float(np.max(np.abs(mapped-edge))),
            "inverse_error_px": inverse_error, "pose_or_model_used": False})
    return regions, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "database", "initial-frames", "reference-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--feature-storage", choices=["eager", "lazy"], default="eager")
    parser.add_argument("--pnp-threads", type=int, choices=[1, 8], default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    helpers = root/"02_code/box2_lio_guided_sfm_v1_20261002"
    sys.path.insert(0, str(helpers))
    common = importlib.import_module("common")
    from feature_localization import FeatureBank, estimate_pose
    from localize_extra_frames import split_correspondences
    admission = read(args.export/"TRAINING_ADMISSION.json")
    if admission["status"] != "export_and_admission_verified_training_pending":
        raise ValueError("New-map development localization requires verified training export")
    model_path = Path(admission["training_opencv_model"])
    model = pycolmap.Reconstruction(model_path)
    pinhole_model = pycolmap.Reconstruction(Path(admission["dataset"])/"sparse/0")
    if len(model.cameras) != 1 or len(pinhole_model.cameras) != 1:
        raise ValueError("This fixed single-camera protocol cannot silently handle multiple calibrations")
    camera, pinhole = next(iter(model.cameras.values())), next(iter(pinhole_model.cameras.values()))
    split_path = args.reference_run/"00_contract/split_manifest.json"
    original_localization_path = args.reference_run/"05_evaluation/C_augmented_212/H_LOCALIZATION.json"
    split = read(split_path)
    evaluation_rows = [row for row in split["extra_frames"] if row["role"] == "H"]
    old_regions = read(original_localization_path)
    old_intrinsics = old_regions["PINHOLE_camera"]
    old_camera_db = args.reference_run/"00_contract/database.db"
    with sqlite3.connect(old_camera_db.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        old_camera_rows = connection.execute("SELECT camera_id,model,width,height,params FROM cameras").fetchall()
    if len(old_camera_rows) != 1:
        raise ValueError("Old image-region calibration must have one explicit camera")
    cid, mid, width, height, values = old_camera_rows[0]
    old_raw = pycolmap.Camera(camera_id=cid, model=pycolmap.CameraModelId(mid), width=width, height=height,
                             params=np.frombuffer(values, np.float64).copy())
    old_pinhole = pycolmap.Camera(model="PINHOLE", width=old_intrinsics["width"], height=old_intrinsics["height"],
                                  params=old_intrinsics["params"])
    if (old_raw.width, old_raw.height) != (camera.width, camera.height):
        raise ValueError("Old and new calibrations refer to different raw sensor image dimensions")
    regions, region_checks = transfer_regions(old_regions["rows"], old_pinhole, old_raw, camera, pinhole)
    atomic_json(args.output/"REGION_TRANSFER.json", {"status": "calibration_transfer_verified",
        "old_raw_camera": common.serializable(old_raw.todict()), "new_raw_camera": common.serializable(camera.todict()),
        "old_reference_camera": old_intrinsics, "new_reference_camera": common.serializable(pinhole.todict()),
        "old_camera_metadata_source": str(old_camera_db), "rows": region_checks,
        "boundary": "shared raw pixels, nearest-integer rectangle edges; no old camera pose or model geometry"})
    if len(evaluation_rows) != 24 or {row["image_name"] for row in evaluation_rows} != set(regions):
        raise ValueError("The original complete 24-view denominator changed")
    raw_path = args.initial_frames/"frame_decisions.jsonl"
    raw = {row["image_name"]: row for row in map(json.loads, raw_path.read_text(encoding="utf-8-sig").splitlines())}
    training_names = {im.name for im in model.images.values() if im.has_pose}
    excluded = {name for name, row in raw.items() if row["excluded_role"]}
    if len(excluded) != 48 or training_names & excluded:
        raise ValueError("Protected development images entered the training map")
    source_paths = sorted(model_path.glob("*.bin")) + [args.database, raw_path, split_path]
    identities = [data_identity(path) for path in source_paths]
    protocol = {"status": "frozen_before_new_map_localization", "input_version": "new_map_admission_v1",
        "feature_storage": args.feature_storage,
        "PnP_threads": args.pnp_threads,
        "feature_store_source": data_identity(Path(__file__).with_name("sift_feature_store.py")),
        "maximum_seconds": 900, "maximum_RSS_bytes": 32*1024**3, "all_H_count": 24,
        "protected_development_exclusion_count": 48,
        "source_identity": identities, "training_admission": data_identity(args.export/"TRAINING_ADMISSION.json"),
        "source": data_identity(Path(__file__)),
        "helpers": [data_identity(helpers/name) for name in ("feature_localization.py", "localize_extra_frames.py")],
        "SIFT_max_features": 12000, "SIFT_peak_threshold": .004,
        "ratio_max": .8, "normalized_distance_max": .7, "mutual_check": True,
        "PnP_min_inliers": 30, "PnP_max_error_px": 4.,
        "neighbors": "four temporal plus eight raw-position/heading neighbors",
        "H_loc": "even features and their geometric verification only",
        "H_eval": "odd tentative matches excluding every localization 3D point",
        "render_ROI": "original frozen rectangles transferred through sensor pixels using only calibrations; no old world-space ROI or poses imported",
        "rectangle_source": data_identity(original_localization_path),
        "old_pose_values_used": False, "old_3D_points_used": False, "map_update": False,
        "strict_blind_test": False, "independent_pose_accuracy_claim": False}
    atomic_json(args.output/"LOCALIZATION_CONTRACT.json", protocol)
    state = {"status": "loading_readonly_feature_database", "completed_frames": 0}
    atomic_json(args.output/"STATE.json", state)
    results = []
    lazy_store = None
    original_pose_estimator = pycolmap.estimate_and_refine_absolute_pose
    def selected_pose_estimator(xy, xyz, calibration, options, refinement):
        if options.ransac.num_threads != 8:
            raise ValueError("The original PnP options differ from the recorded contract")
        options.ransac.num_threads = args.pnp_threads
        return original_pose_estimator(xy, xyz, calibration, options, refinement)
    try:
        pycolmap.estimate_and_refine_absolute_pose = selected_pose_estimator
        feature_load_started = time.monotonic()
        bank = FeatureBank.__new__(FeatureBank)
        bank.run, bank.protocol = args.output, protocol
        cv2.setNumThreads(8)
        options = pycolmap.FeatureExtractionOptions()
        options.sift.max_num_features, options.sift.peak_threshold = 12000, .004
        bank.extractor = pycolmap.Sift(options, pycolmap.Device.cpu)
        bank.original = {}
        if args.feature_storage == "lazy":
            from sift_feature_store import SiftFeatureStore, metadata_neighbors
            lazy_store = SiftFeatureStore(args.database, training_names, "lazy", 32)
            if any(i not in model.images or model.images[i].name != name for i, name in lazy_store.names.items()):
                raise ValueError("Lazy feature metadata differs from the registered map")
            bank.original = lazy_store
            bank.neighbors = lambda rec, rough, stamp: metadata_neighbors(lazy_store, rec, rough, stamp)
        else:
            connection = sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro", uri=True)
            try:
                query = "SELECT images.image_id,images.name,keypoints.rows,keypoints.cols,keypoints.data,descriptors.rows,descriptors.cols,descriptors.data FROM images JOIN keypoints USING(image_id) JOIN descriptors USING(image_id)"
                for image_id, name, kr, kc, kd, dr, dc, dd in connection.execute(query):
                    if image_id not in model.images or not model.images[image_id].has_pose:
                        continue
                    if kr != dr or dc != 128 or model.images[image_id].name != name:
                        raise ValueError("Database feature and camera identities differ from new map")
                    bank.original[int(image_id)] = (name, np.frombuffer(kd, np.float32).reshape(kr, kc).copy(),
                        np.frombuffer(dd, np.uint8).reshape(dr, dc).astype(np.float32)/512.)
            finally:
                connection.close()
        if len(bank.original) != model.num_reg_images():
            raise ValueError("A registered training camera is missing its frozen features")
        feature_load_seconds = time.monotonic()-feature_load_started
        feature_RSS = psutil.Process().memory_info().rss
        feature_timing = {"mode": args.feature_storage, "load_seconds": feature_load_seconds,
            "RSS_after_load_bytes": feature_RSS, "metadata_training_cameras": len(bank.original),
            "resident_training_cameras_after_load": lazy_store.receipt()["cached_images"] if lazy_store is not None else len(bank.original),
            "resident_array_bytes": lazy_store.receipt()["cached_array_bytes"] if lazy_store is not None else
                sum(k.nbytes+d.nbytes for _, k, d in bank.original.values()),
            "rows": []}
        access_trace = []
        bank.cache = args.output/"feature_cache"
        bank.cache.mkdir()
        references = args.output/"reference_H"
        references.mkdir()
        original_parameters = camera.params.copy()
        for source in evaluation_rows:
            query_started = time.monotonic()
            name = source["image_name"]
            rough = raw[name]
            if rough["excluded_role"] != "H" or rough["camera_header_ns"] != source["camera_header_ns"]:
                raise ValueError("Raw timestamp or evaluation role identity changed")
            image_path = args.reference_run/"00_contract/extra_images"/name
            with Image.open(image_path) as photo:
                bitmap, reference_camera = pycolmap.undistort_image(pycolmap.UndistortCameraOptions(),
                    pycolmap.Bitmap.from_array(np.asarray(photo.convert("RGB"))), camera)
            if (not np.array_equal(reference_camera.params, pinhole.params)
                    or (reference_camera.width, reference_camera.height) != (pinhole.width, pinhole.height)):
                raise ValueError("Evaluation and training undistortion differ")
            reference = references/(Path(name).stem+".png")
            Image.fromarray(bitmap.to_array()).save(reference)
            base = {**source, "image_reference": str(reference.resolve()),
                "raw_image": data_identity(image_path), "target_window_PINHOLE": regions[name],
                "PnP_uses_even_features_only": True, "map_updated": False}
            if not rough["coarse_pose_valid"]:
                record = {**base, "status": "failed_missing_raw_pose_neighbor_support"}
            else:
                keypoints, descriptors = bank.extra(image_path)
                neighbors = bank.neighbors(model, rough, source["camera_header_ns"])
                access_trace.extend(neighbors)
                access_trace.extend(neighbors)
                correspondence, loc_edges, eval_edges = split_correspondences(bank, model, keypoints, descriptors, neighbors)
                solution, info, loc, unused = estimate_pose(model, keypoints, correspondence, split=True)
                record = {**base, **info, "feature_count": len(keypoints),
                    "localization_edges": loc_edges, "evaluation_edges": eval_edges}
                if solution is not None and info["loc_eval_3D_overlap"] != 0:
                    raise ValueError("Development localization and diagnostic matches share 3D points")
            results.append(record)
            feature_timing["rows"].append({"image_name": name, "total_query_seconds": time.monotonic()-query_started})
            memory = psutil.Process().memory_info()
            peak = max(memory.rss, getattr(memory, "peak_wset", memory.rss))
            state.update(status="localizing", completed_frames=len(results), last_frame=name,
                last_frame_status=record["status"], elapsed_seconds=time.monotonic()-started, peak_RSS_bytes=peak)
            atomic_json(args.output/"STATE.json", state)
            atomic_json(args.output/"PARTIAL_RESULTS.json", common.serializable(results))
            print(json.dumps({"frame": source["camera_frame"], "status": record["status"]}), flush=True)
            if peak > 32*1024**3 or time.monotonic()-started > 900:
                raise RuntimeError("The registered localization resource budget was exceeded")
        if not np.array_equal(original_parameters, camera.params) or identities != [data_identity(path) for path in source_paths]:
            raise ValueError("Read-only localization changed a frozen source or camera calibration")
        count = sum(row["status"] == "localized" for row in results)
        result = {"status": "all24_localized_and_frozen" if count == 24 else "failed_complete_localization_guard",
            "input_version": "new_map_admission_v1", "summary": {"all_H_count": 24, "localized_count": count,
                "all_H_denominator_kept": True, "quality_eligible_H_count": sum(r["quality_eligible"] for r in results)},
            "rows": results, "PINHOLE_camera": common.serializable(pinhole.todict()),
            "map_source": str(model_path), "map_updated": False, "H_data_used_for_selection": False,
            "strict_blind_test": False, "independent_pose_accuracy_claim": False,
            "source_unchanged": True, "elapsed_seconds": time.monotonic()-started}
        feature_timing["final_store"] = lazy_store.receipt() if lazy_store is not None else None
        atomic_json(args.output/"FEATURE_COST.json", feature_timing)
        atomic_json(args.output/"FEATURE_ACCESS_TRACE.json", {"status": "observed_actual_match_requests",
            "database": data_identity(args.database), "registered_names": sorted(training_names),
            "access_image_ids": access_trace, "query_count": 24,
            "role": "matching access only; neighbor-name metadata reads do not require descriptor payload"})
        atomic_json(args.output/"H_LOCALIZATION.json", common.serializable(result))
        state.update(status=result["status"], localized_count=count)
        atomic_json(args.output/"STATE.json", state)
        if count != 24:
            raise RuntimeError("Incomplete localization retains all failures and cannot pass rendering evaluation")
    except BaseException as error:
        state.update(status="failed_localization_no_quality_acceptance", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        pycolmap.estimate_and_refine_absolute_pose = original_pose_estimator
        if lazy_store is not None:
            lazy_store.close()


if __name__ == "__main__":
    main()
