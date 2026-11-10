"""Consume owned exact match arrays without changing native geometry or ordering."""

from contextlib import contextmanager
import time

import cv2
import numpy as np

from integer_sift_matching import ExactMatcher


class ArrayMatchGeometry:
    """Keep caller RNG, sequential error order, and full tuple tie breaking."""

    def __init__(self, original, enabled=True):
        self.original, self.enabled = original, enabled
        self.trace = {"routes": [], "array_neighbors": 0, "counted": False}

    def __call__(self, bank, model, keypoints, descriptors, neighbors, index, geometric=True):
        compatible = (self.enabled and cv2.__version__ == "4.13.0"
            and isinstance(geometric, bool) and isinstance(neighbors, (list, tuple))
            and len(neighbors) <= 12 and isinstance(descriptors, np.ndarray)
            and descriptors.dtype == np.float32 and descriptors.ndim == 2
            and descriptors.shape[1] == 128 and len(descriptors) <= 12000
            and isinstance(keypoints, np.ndarray) and keypoints.ndim == 2
            and keypoints.shape[1] >= 2 and len(keypoints) == len(descriptors)
            and keypoints.dtype in (np.float32, np.float64))
        if not compatible:
            self.trace["routes"].append("original_before_consumption")
            return self.original(bank, model, keypoints, descriptors, neighbors, index, geometric)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        if type(matcher) is not ExactMatcher or matcher.pending is not None:
            self.trace["routes"].append("original_before_consumption")
            return self.original(bank, model, keypoints, descriptors, neighbors, index, geometric)
        self.trace["routes"].append("owned_exact_array")
        correspondences, edges, all_pairs = [], [], []
        for image_id in neighbors:
            name, train_keypoints, train_descriptors = bank.original[image_id]
            pixels, point_ids, valid = index.get(model, image_id)
            if len(train_keypoints) != len(pixels):
                raise ValueError("Original keypoint and map observation identity mismatch")
            if not np.allclose(train_keypoints[:, :2], pixels, atol=1e-3):
                raise ValueError("Original SIFT pixels and map pixels disagree")
            if not matcher.prepared:
                raise ValueError("A native neighbor consumed more exact batches than registered")
            expected_query, expected_train, values = matcher.prepared.pop(0)
            if not np.array_equal(descriptors, expected_query) or not np.array_equal(train_descriptors, expected_train):
                raise ValueError("The exact neighbor batch order or native descriptor values changed")
            forward, distances, reverse = values
            d = distances.astype(np.float64)
            mask = ((d[:, 0] < .8*d[:, 1]) & (d[:, 0] <= .7)
                & (reverse[forward[:, 0]] == np.arange(len(descriptors))))
            selected = np.flatnonzero(mask)
            self.trace["array_neighbors"] += 1
            raw_count, matrix = len(selected), None
            if geometric and len(selected) >= 15:
                xy1 = np.asarray(keypoints[selected, :2])
                xy2 = np.asarray(train_keypoints[forward[selected, 0], :2])
                cv2.setRNGSeed(20261002)
                matrix, inliers = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, 4., .999, 10000)
                selected = np.empty(0, np.intp) if matrix is None or inliers is None else np.asarray(
                    [i for i, ok in zip(selected, inliers.ravel()) if ok], dtype=np.intp)
            if geometric and len(selected) < 15:
                selected = np.empty(0, np.intp)
            edges.append({"train_image_id": image_id, "train_image_name": name,
                "mutual_ratio_matches": raw_count, "geometric_inliers": len(selected),
                "F_from_query_localization_features": None if matrix is None else matrix.tolist()})
            for row in selected:
                query, train, distance = int(row), int(forward[row, 0]), float(d[row, 0])
                all_pairs.append((image_id, query, train, distance))
                if valid[train]:
                    correspondences.append((distance, query, int(point_ids[train]), image_id, train))
        unique, used_query, used_point = [], set(), set()
        for distance, query, point_id, image_id, train_index in sorted(correspondences):
            if query in used_query or point_id in used_point:
                continue
            used_query.add(query)
            used_point.add(point_id)
            unique.append({"query": query, "point_id": point_id, "train_image_id": image_id,
                "train_index": train_index, "distance": float(distance)})
        return unique, edges, all_pairs


@contextmanager
def array_match_geometry(enabled=True):
    """Restore the private entry hook on normal, rejected, and error paths."""
    import native_observation_index
    started = time.perf_counter()
    original = native_observation_index.indexed_match
    operator = ArrayMatchGeometry(original, enabled)
    native_observation_index.indexed_match = operator
    operator.trace["setup_seconds"] = time.perf_counter()-started
    try:
        yield operator.trace
    finally:
        started = time.perf_counter()
        native_observation_index.indexed_match = original
        operator.trace["binding_restored"] = native_observation_index.indexed_match is original
        operator.trace["shutdown_seconds"] = time.perf_counter()-started
