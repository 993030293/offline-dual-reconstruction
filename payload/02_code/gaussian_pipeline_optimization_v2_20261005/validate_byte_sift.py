"""Verify real byte-SIFT values, native L2 matches, rejected input, and restoration."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002")]
from runtime import atomic_json, data_identity
from byte_sift_matching import TransientFloatView, byte_sift_matching
import feature_localization
import sift_feature_store


def match_values(matches):
    return [(m.queryIdx, m.trainIdx, m.imgIdx, m.distance) for m in matches]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "database", "query-features"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, details = {}, {}
    with np.load(args.query_features, allow_pickle=False) as features:
        query = features["descriptors"].copy()
    original_query = query.copy()
    checks["actual_query_is_original_float32"] = query.dtype == np.float32 and query.ndim == 2 and query.shape[1] == 128
    connection = sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro", uri=True)
    records = connection.execute("SELECT images.image_id,images.name,keypoints.rows,keypoints.cols,keypoints.data,descriptors.rows,descriptors.cols,descriptors.data FROM images JOIN keypoints USING(image_id) JOIN descriptors USING(image_id) ORDER BY image_id LIMIT 4").fetchall()
    connection.close()
    cv2.setNumThreads(8)
    original_decode = sift_feature_store.SiftFeatureStore.decode
    original_match = feature_localization.FeatureBank.match
    original_matcher = cv2.BFMatcher
    class StoreFixture:
        def __init__(self):
            self.names = {int(r[0]): r[1] for r in records}
            self.records = {int(r[0]): r[1:] for r in records}

        def __getitem__(self, image_id):
            return sift_feature_store.SiftFeatureStore.decode(self, image_id, self.records[image_id])

    store = StoreFixture()
    original_arrays = {int(r[0]): original_decode(store, int(r[0]), r[1:]) for r in records}
    checks["four_actual_training_records"] = len(records) == 4
    with byte_sift_matching(True) as trace:
        view = TransientFloatView(store, trace)
        checks["original_cv2_native_matcher_unchanged"] = cv2.BFMatcher is original_matcher
        for record in records:
            image_id = int(record[0])
            name, keypoints, train = sift_feature_store.SiftFeatureStore.decode(store, image_id, record[1:])
            old_name, old_keypoints, old_train = original_arrays[image_id]
            checks[f"training_{image_id}_record_identity_exact"] = name == old_name and np.array_equal(keypoints, old_keypoints)
            checks[f"training_{image_id}_descriptor_values_exact"] = np.array_equal(old_train, train.astype(np.float32)/512.)
            _, _, restored_train = view[image_id]
            checks[f"training_{image_id}_matcher_float_values_exact"] = np.array_equal(restored_train, old_train)
            float_matcher = original_matcher(cv2.NORM_L2)
            candidate_matcher = cv2.BFMatcher(cv2.NORM_L2)
            float_forward = float_matcher.knnMatch(query, old_train, k=2)
            byte_forward = candidate_matcher.knnMatch(query, restored_train, k=2)
            checks[f"training_{image_id}_all_knn_matches_exact"] = [match_values(r) for r in float_forward] == [match_values(r) for r in byte_forward]
            float_reverse = float_matcher.match(old_train, query)
            byte_reverse = candidate_matcher.match(restored_train, query)
            checks[f"training_{image_id}_all_mutual_matches_exact"] = match_values(float_reverse) == match_values(byte_reverse)
            details[str(image_id)] = {"training_descriptor_rows": len(train), "query_rows": len(query),
                "float_array_bytes": old_train.nbytes, "byte_array_bytes": train.nbytes}
        checks["all_query_values_unchanged_after_real_matching"] = np.array_equal(query, original_query)
        bank = SimpleNamespace(original=store)
        try:
            feature_localization.FeatureBank.match(bank, None, np.zeros((len(query), 4), np.float32), query, [int(records[0][0])])
            checks["matching_error_restores_original_store"] = False
        except AttributeError:
            checks["matching_error_restores_original_store"] = bank.original is store
        for name, invalid_query in (("float64", query.astype(np.float64)), ("wrong_dimension", query[:, :127]),
            ("NaN", np.full((1, 128), np.nan, np.float32))):
            try:
                feature_localization.FeatureBank.match(bank, None, np.zeros((len(invalid_query), 4), np.float32), invalid_query, [])
                checks[name+"_rejected"] = False
            except ValueError:
                checks[name+"_rejected"] = bank.original is store
    checks["normal_context_restores_all_patched_callables"] = (sift_feature_store.SiftFeatureStore.decode is original_decode and
        feature_localization.FeatureBank.match is original_match and cv2.BFMatcher is original_matcher)
    try:
        with byte_sift_matching(True):
            raise RuntimeError("Intentional byte-matching restoration check")
    except RuntimeError:
        pass
    checks["error_context_restores_all_patched_callables"] = (sift_feature_store.SiftFeatureStore.decode is original_decode and
        feature_localization.FeatureBank.match is original_match and cv2.BFMatcher is original_matcher)
    sources = [Path(__file__), Path(__file__).with_name("byte_sift_matching.py"),
        Path(feature_localization.__file__), Path(sift_feature_store.__file__)]
    result = {"status": "lifecycle_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "details": details, "sources": [data_identity(p) for p in sources],
        "query_features": data_identity(args.query_features), "database": data_identity(args.database),
        "counted": False, "not_a_whole_route_speed_or_quality_result": True}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks}), flush=True)
    if not all(checks.values()):
        raise ValueError("Real exact byte-SIFT lifecycle did not pass")


if __name__ == "__main__":
    main()
