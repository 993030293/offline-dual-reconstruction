"""Verify real 24-view matches, neighbors and PnP before feature-store acceptance."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import pycolmap

from runtime import atomic_json, data_identity
from sift_feature_store import SiftFeatureStore, metadata_neighbors


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "resources", "database", "initial-frames", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    helpers = Path(__file__).resolve().parents[2]/"02_code/box2_lio_guided_sfm_v1_20261002"
    sys.path.insert(0, str(helpers))
    from feature_localization import FeatureBank, estimate_pose
    from localize_extra_frames import split_correspondences
    cv2.setNumThreads(8)
    baseline, candidate = [read(path/"H_LOCALIZATION.json") for path in (args.baseline, args.candidate)]
    resource = read(args.resources/"STATE.json")
    if (resource["status"] != "resource_and_equivalence_passed_real_matching_pending"
            or baseline["status"] != "all24_localized_and_frozen" or candidate["status"] != "all24_localized_and_frozen"):
        raise ValueError("All independent resource and real localization prerequisites must pass")
    pairs = resource["pairs"]
    rss = {mode: statistics.median(p[mode]["RSS_after_access_bytes"] for p in pairs) for mode in ("eager", "lazy")}
    times = {mode: statistics.median(p[mode]["load_and_access_seconds"] for p in pairs) for mode in rss}
    checks = {"three_resource_pairs": len(pairs) == 3,
        "all_resource_arrays_exact": all(p["eager"]["rows"] == p["lazy"]["rows"] for p in pairs),
        "RSS_reduction_at_least20percent": 1-rss["lazy"]/rss["eager"] >= .2,
        "time_ratio_at_most1_10": times["lazy"]/times["eager"] <= 1.10,
        "bounded32_cache": all(p["lazy"]["store"]["maximum_cached_images"] <= 32 for p in pairs),
        "source_matches_resource_trials": all(data_identity(Path(__file__).with_name(Path(r["path"]).name))["sha256"] == r["sha256"]
            for r in resource["sources"] if Path(r["path"]).name == "sift_feature_store.py"),
        "same24_evaluation_names": len(baseline["rows"]) == len(candidate["rows"]) == 24 and
            [r["image_name"] for r in baseline["rows"]] == [r["image_name"] for r in candidate["rows"]],
        "same_new_map": baseline["map_source"] == candidate["map_source"],
        "same_full_intrinsics": baseline["PINHOLE_camera"] == candidate["PINHOLE_camera"],
        "same_PnP_thread_setting": read(args.baseline/"LOCALIZATION_CONTRACT.json").get("PnP_threads",8)
            == read(args.candidate/"LOCALIZATION_CONTRACT.json").get("PnP_threads",8)}
    model = pycolmap.Reconstruction(baseline["map_source"])
    registered = {im.name for im in model.images.values() if im.has_pose}
    raw = {r["image_name"]: r for r in map(json.loads, (args.initial_frames/"frame_decisions.jsonl").read_text().splitlines())}
    database_identity = data_identity(args.database)
    stores = {mode: SiftFeatureStore(args.database, registered, mode, 32) for mode in ("eager", "lazy")}
    banks = {}
    for mode, store in stores.items():
        bank = FeatureBank.__new__(FeatureBank)
        bank.original = store
        banks[mode] = bank
    records = []
    try:
        for left, right in zip(baseline["rows"], candidate["rows"]):
            name = left["image_name"]
            source = args.baseline/"feature_cache"/(Path(name).stem+".npz")
            alternate = args.candidate/"feature_cache"/source.name
            with np.load(source) as archive, np.load(alternate) as other:
                keypoints, descriptors = archive["keypoints"].copy(), archive["descriptors"].copy()
                query_equal = np.array_equal(keypoints, other["keypoints"]) and np.array_equal(descriptors, other["descriptors"])
            normal = banks["eager"].neighbors(model, raw[name], left["camera_header_ns"])
            lazy = metadata_neighbors(stores["lazy"], model, raw[name], left["camera_header_ns"])
            normal_matches = split_correspondences(banks["eager"], model, keypoints, descriptors, normal)
            lazy_matches = split_correspondences(banks["lazy"], model, keypoints, descriptors, lazy)
            row = {"image_name": name, "query_arrays_exact": query_equal, "neighbors_exact": normal == lazy,
                "all_correspondence_ids_and_distances_exact": normal_matches[0] == lazy_matches[0],
                "all_localization_geometry_exact": normal_matches[1] == lazy_matches[1],
                "all_evaluation_tentative_matches_exact": normal_matches[2] == lazy_matches[2],
                "real_localization_edges_exact": left["localization_edges"] == right["localization_edges"],
                "real_evaluation_edges_exact": left["evaluation_edges"] == right["evaluation_edges"],
                "same_target_window": left["target_window_PINHOLE"] == right["target_window_PINHOLE"],
                "same_reference_image_bytes": data_identity(Path(left["image_reference"]))["sha256"] == data_identity(Path(right["image_reference"]))["sha256"],
                "same_actual_PnP_inliers": left["inliers"] == right["inliers"]}
            pose_error = max(float(np.max(np.abs(np.asarray(left[key])-np.asarray(right[key])))) for key in ("R_CW", "t_CW"))
            row["actual_PnP_pose_within_original1e_9"] = pose_error <= 1e-9
            row["maximum_pose_element_delta"] = pose_error
            row["passed"] = all(value for key, value in row.items() if isinstance(value, bool))
            records.append(row)
            atomic_json(args.output/"STATE.json", {"status": "matching_checks_running", "completed_views": len(records), "last": row})
        checks["all24_actual_matching_and_localization_checks"] = len(records) == 24 and all(r["passed"] for r in records)
        checks["database_unchanged"] = database_identity == data_identity(args.database)
        checks["real_entry_uses_bounded_lazy_storage"] = read(args.candidate/"FEATURE_COST.json")["final_store"]["maximum_cached_images"] <= 32
        result = {"status": "accepted_stage_optimization" if all(checks.values()) else "failed_stage_acceptance",
            "mechanism": "P27_LAZY_SIFT_FEATURES", "checks": checks, "records": records,
            "median_RSS_bytes": rss, "RSS_reduction": 1-rss["lazy"]/rss["eager"],
            "median_load_and_access_seconds": times, "median_time_ratio": times["lazy"]/times["eager"],
            "resources": data_identity(args.resources/"STATE.json"),
            "real_localization_inputs": [data_identity(p/"H_LOCALIZATION.json") for p in (args.baseline,args.candidate)],
            "source": data_identity(Path(__file__)), "counted": all(checks.values()),
            "benefit": "bounded host feature-memory footprint with identical real matching and poses",
            "total_SfM_speed_claim": False, "absolute_pose_accuracy_improved": False,
            "full_route_combination_accepted": False, "elapsed_seconds": time.monotonic()-started}
        atomic_json(args.output/"ACCEPTANCE.json", result)
        atomic_json(args.output/"STATE.json", {"status": result["status"], "counted": result["counted"]})
        print(json.dumps({"status": result["status"], "RSS_reduction": result["RSS_reduction"], "checks": checks}))
        if not all(checks.values()):
            raise ValueError("A frozen feature-store acceptance guard failed")
    finally:
        for store in stores.values():
            store.close()


if __name__ == "__main__":
    main()
