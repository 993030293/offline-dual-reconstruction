"""Localize every frozen raw evaluation frame in its own freshly reconstructed map."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import psutil
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
HELPERS = ROOT / "02_code/box2_lio_guided_sfm_v1_20261002"
sys.path[:0] = [str(V1), str(HELPERS)]
from runtime import atomic_json, data_identity
from feature_localization import FeatureBank, estimate_pose
from localize_extra_frames import split_correspondences
from sift_feature_store import SiftFeatureStore, metadata_neighbors


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    admission = read(args.side / "export/TRAINING_ADMISSION.json")
    if admission["status"] != "export_and_admission_verified_training_pending":
        raise ValueError("Evaluation requires the verified fresh training subset")
    model_path = Path(admission["training_opencv_model"])
    model = pycolmap.Reconstruction(model_path)
    pinhole = next(iter(pycolmap.Reconstruction(Path(admission["dataset"]) / "sparse/0").cameras.values()))
    if len(model.cameras) != 1 or pinhole.model_name != "PINHOLE":
        raise ValueError("The evaluation protocol requires one fixed calibrated sensor")
    camera = next(iter(model.cameras.values()))
    database = next((args.side / label / "database.db" for label in ("pose_gap", "temporal", "recovery", "recovery_not_triggered")
                     if (args.side / label / "database.db").exists()), None)
    if database is None:
        raise FileNotFoundError("The fresh source feature database is unavailable")
    evaluation = read(args.side / "initial/EVALUATION_FRAMES.json")["rows"]
    if len(evaluation) != 24 or len({row["image_name"] for row in evaluation}) != 24:
        raise ValueError("The pre-selection complete raw evaluation denominator changed")
    training_names = {image.name for image in model.images.values() if image.has_pose}
    raw_rows = [json.loads(line) for line in (args.side / "initial/frame_decisions.jsonl").read_text().splitlines()]
    exclusions = {row["image_name"] for row in raw_rows if row["excluded_role"]}
    if training_names & exclusions or len(exclusions) != 48:
        raise ValueError("Protected evaluation or adjacent exclusion frames entered training")
    manifest = read(args.side / "initial/INITIAL_FRAME_MANIFEST.json")
    policy = manifest["policy"]
    sharpness = manifest["receipt"]["sharpness_min"]
    sources = [model_path / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    sources.extend([database, args.side / "initial/EVALUATION_FRAMES.json"])
    identities = [data_identity(path) for path in sources]
    contract = {"status": "frozen_before_first_localization", "mode": args.mode,
        "source_identity": identities, "all_evaluation_count": 24, "PnP_threads": 1, "SIFT_and_OpenCV_CPU_threads": 4,
        "SIFT_max_features": 12000, "SIFT_peak_threshold": .004, "neighbors": "four temporal and eight coarse-pose neighbors",
        "localization": "even query features only; odd evaluation matches exclude all localization 3D IDs",
        "map_updated": False, "old_SfM_or_poses_used": False, "GT_used": False,
        "strict_blind_test": False, "source": data_identity(Path(__file__)),
        "quality_region": "whole image primary; fixed central half-width half-height diagnostic only",
        "central_region_is_not_object_ground_truth": True}
    atomic_json(args.output / "CONTRACT.json", contract)
    state = {"status": "loading_features", "completed_frames": 0, "pid": psutil.Process().pid}
    atomic_json(args.output / "STATE.json", state)
    bank = FeatureBank.__new__(FeatureBank)
    bank.run, bank.protocol = args.output, contract
    cv2.setNumThreads(4)
    options = pycolmap.FeatureExtractionOptions()
    options.num_threads = 4
    options.sift.max_num_features, options.sift.peak_threshold = 12000, .004
    bank.extractor = pycolmap.Sift(options, pycolmap.Device.cpu)
    store = SiftFeatureStore(database, training_names, "eager" if args.mode == "baseline" else "lazy", 32)
    bank.original = store
    if any(image_id not in model.images or model.images[image_id].name != name for image_id, name in store.names.items()):
        raise ValueError("SIFT metadata and fresh model identities differ")
    if args.mode == "optimized":
        bank.neighbors = lambda rec, rough, stamp: metadata_neighbors(store, rec, rough, stamp)
    bank.cache = args.output / "feature_cache"
    bank.cache.mkdir()
    references = args.output / "reference_images"
    references.mkdir()
    store_load_seconds = time.monotonic() - started
    store_initial = store.receipt()
    native_estimator = pycolmap.estimate_and_refine_absolute_pose

    def estimate(xy, xyz, calibration, estimation, refinement):
        if estimation.ransac.num_threads != 8:
            raise ValueError("The preserved helper's estimator options changed")
        estimation.ransac.num_threads = 1
        return native_estimator(xy, xyz, calibration, estimation, refinement)

    pycolmap.estimate_and_refine_absolute_pose = estimate
    parameters = camera.params.copy()
    results, requests = [], []
    try:
        for source in evaluation:
            before = time.monotonic()
            name = source["image_name"]
            image_path = args.side / "initial/evaluation_images" / name
            with Image.open(image_path) as photo:
                bitmap, calibrated = pycolmap.undistort_image(pycolmap.UndistortCameraOptions(),
                    pycolmap.Bitmap.from_array(np.asarray(photo.convert("RGB"))), camera)
            if ((calibrated.width, calibrated.height) != (pinhole.width, pinhole.height)
                    or not np.array_equal(calibrated.params, pinhole.params)):
                raise ValueError("Training and evaluation undistortion use different intrinsics")
            reference = references / (Path(name).stem + ".png")
            Image.fromarray(bitmap.to_array()).save(reference)
            quality = source["quality"]
            eligible = (quality["sharpness"] >= sharpness and quality["clipped_fraction"] <= policy["clipped_fraction_max"]
                and policy["brightness_min"] <= quality["mean_brightness"] <= policy["brightness_max"]
                and quality["corner_count_half_resolution"] >= policy["corners_min"]
                and quality["occupied_grid_cells_4x4"] >= policy["occupied_cells_min"])
            row = {"image_name": name, "camera_frame": source["camera_frame"],
                "camera_header_ns": source["camera_header_ns"], "quality_eligible": eligible,
                "image_reference": str(reference.resolve()), "raw_image": data_identity(image_path),
                "central_window_PINHOLE": [round(pinhole.width*.25), round(pinhole.height*.25),
                                           round(pinhole.width*.75), round(pinhole.height*.75)]}
            if not source["coarse_pose_valid"]:
                row["status"] = "failed_missing_coarse_neighbor_support"
            else:
                keypoints, descriptors = bank.extra(image_path)
                neighbors = bank.neighbors(model, source, source["camera_header_ns"])
                requests.extend(neighbors)
                correspondences, loc_edges, eval_edges = split_correspondences(bank, model, keypoints, descriptors, neighbors)
                solution, info, used, unused = estimate_pose(model, keypoints, correspondences, split=True)
                row.update(info, feature_count=len(keypoints), localization_edges=loc_edges, evaluation_edges=eval_edges)
                if solution is not None and info.get("loc_eval_3D_overlap", 0) != 0:
                    raise ValueError("Unused diagnostic matches overlap localization 3D IDs")
            row["elapsed_seconds"] = time.monotonic() - before
            results.append(row)
            state.update(status="localizing", completed_frames=len(results), last_status=row["status"],
                         elapsed_seconds=time.monotonic()-started, RSS_bytes=psutil.Process().memory_info().rss)
            atomic_json(args.output / "STATE.json", state)
            atomic_json(args.output / "PARTIAL_RESULTS.json", results)
            print(json.dumps({"frame": row["camera_frame"], "status": row["status"]}), flush=True)
        if not np.array_equal(parameters, camera.params) or identities != [data_identity(path) for path in sources]:
            raise ValueError("Read-only localization changed source model, database or calibration")
        count = sum(row["status"] == "localized" for row in results)
        result = {"status": "all24_localized" if count == 24 else "partial_localization_complete_guard_failed",
            "input_version": args.side.parent.name + "_fresh_raw_20261008", "rows": results,
            "PINHOLE_camera": {"width": pinhole.width, "height": pinhole.height, "params": pinhole.params.tolist()},
            "summary": {"all_count": 24, "localized_count": count, "failed_count": 24-count,
                        "all_failures_retained": True}, "map_source": str(model_path),
            "map_updated": False, "independent_pose_accuracy_claim": False, "strict_blind_test": False,
            "elapsed_seconds": time.monotonic()-started, "source_unchanged": True}
        atomic_json(args.output / "LOCALIZATION.json", result)
        atomic_json(args.output / "FEATURE_COST.json", {"load_seconds": store_load_seconds,
            "initial_store": store_initial, "final_store": store.receipt(), "access_ids": requests,
            "lazy_feature_store_primary_benefit": "descriptor resident memory; speed independently measured"})
        state.update(status=result["status"], localized_count=count)
        atomic_json(args.output / "STATE.json", state)
    except BaseException as error:
        state.update(status="failed_localization", error=repr(error))
        atomic_json(args.output / "STATE.json", state)
        raise
    finally:
        pycolmap.estimate_and_refine_absolute_pose = native_estimator
        store.close()


if __name__ == "__main__":
    main()
