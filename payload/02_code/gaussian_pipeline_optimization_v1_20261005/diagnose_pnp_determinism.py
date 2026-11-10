"""Repeat unchanged PnP inputs to separate feature equality from solver scheduling."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pycolmap

from runtime import atomic_json, data_identity
from sift_feature_store import SiftFeatureStore, metadata_neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "database", "initial-frames", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    helpers = Path(__file__).resolve().parents[2]/"02_code/box2_lio_guided_sfm_v1_20261002"
    sys.path.insert(0, str(helpers))
    from feature_localization import FeatureBank, estimate_pose
    from localize_extra_frames import split_correspondences
    cv2.setNumThreads(8)
    baseline = json.loads((args.baseline/"H_LOCALIZATION.json").read_text())
    model = pycolmap.Reconstruction(baseline["map_source"])
    raw = {r["image_name"]: r for r in map(json.loads, (args.initial_frames/"frame_decisions.jsonl").read_text().splitlines())}
    store = SiftFeatureStore(args.database, {im.name for im in model.images.values() if im.has_pose}, "lazy", 32)
    bank = FeatureBank.__new__(FeatureBank)
    bank.original = store
    original = pycolmap.estimate_and_refine_absolute_pose
    def serialized(xy, xyz, camera, options, refinement):
        if options.ransac.num_threads != 8:
            raise ValueError("The expected native PnP threading contract changed")
        options.ransac.num_threads = 1
        return original(xy, xyz, camera, options, refinement)
    records = []
    try:
        for row in baseline["rows"]:
            if row["camera_frame"] not in (1967, 2136, 2806, 3116, 4509):
                continue
            name = row["image_name"]
            path = args.baseline/"feature_cache"/(Path(name).stem+".npz")
            with np.load(path) as data:
                keypoints, descriptors = data["keypoints"].copy(), data["descriptors"].copy()
            neighbors = metadata_neighbors(store, model, raw[name], row["camera_header_ns"])
            correspondence = split_correspondences(bank, model, keypoints, descriptors, neighbors)[0]
            cases = {}
            for threads in (8, 1):
                pycolmap.estimate_and_refine_absolute_pose = original if threads == 8 else serialized
                values = []
                for _ in range(4):
                    _, info, _, _ = estimate_pose(model, keypoints, correspondence, split=True)
                    values.append(info)
                maximum = max(float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key]))))
                    for a in values for b in values for key in ("R_CW", "t_CW"))
                cases[str(threads)] = {"maximum_repeated_pose_delta": maximum,
                    "inliers": [v["inliers"] for v in values], "all_poses_within1e_9": maximum <= 1e-9}
            records.append({"image_name": name, "query_features": data_identity(path),
                "unchanged_correspondences": len(correspondence), "cases": cases})
        receipt = {"status": "diagnostic_completed", "records": records,
            "native8_nondeterminism_observed": any(not r["cases"]["8"]["all_poses_within1e_9"] for r in records),
            "serialized1_all_repeats_stable": len(records) == 5 and all(r["cases"]["1"]["all_poses_within1e_9"] for r in records),
            "same_arrays_neighbors_correspondences_for_every_repeat": True,
            "source": data_identity(Path(__file__)), "counted": False,
            "elapsed_seconds": time.monotonic()-started}
        atomic_json(args.output/"RESULT.json", receipt)
        print(json.dumps(receipt), flush=True)
    finally:
        pycolmap.estimate_and_refine_absolute_pose = original
        store.close()


if __name__ == "__main__":
    main()
