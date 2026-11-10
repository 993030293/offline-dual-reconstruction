"""Parallelize independent native F fits with ordered assembly and caller RNG parity."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import threading
import time

import cv2
import numpy as np


class ParallelNeighborGeometry:
    def __init__(self, native, parallel, workers=4, maximum_neighbors=12, maximum_array_bytes=64*1024**2):
        started = time.perf_counter()
        if type(parallel) is not bool or not 1 <= workers <= 4 or not 1 <= maximum_neighbors <= 12 or not 1 <= maximum_array_bytes <= 64*1024**2:
            raise ValueError("Neighbor geometry requires explicit bounded scheduling")
        source = inspect.getsource(native)
        for anchor in ("matrix, mask = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, 4., .999, 10000)",
                       "cv2.setRNGSeed(20261002)", "if geometric and len(pairs) >= 15:"):
            if source.count(anchor) != 1:
                raise ValueError("The pinned native geometric estimator changed")
        self.native, self.parallel, self.workers = native, parallel, workers
        self.maximum_neighbors, self.maximum_array_bytes = maximum_neighbors, maximum_array_bytes
        self.pool = None
        self.closed = False
        self.trace = {"routes": [], "geometry_jobs": [], "maximum_neighbors": maximum_neighbors,
            "maximum_array_bytes": maximum_array_bytes, "workers": workers, "peak_pending": 0,
            "caller_thread": threading.get_ident(), "original_last_valid_F_stays_on_caller": True,
            "setup_seconds": time.perf_counter()-started, "shutdown_seconds": 0., "closed": False}

    def fit(self, xy1, xy2):
        started = time.perf_counter()
        cv2.setRNGSeed(20261002)
        matrix, mask = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, 4., .999, 10000)
        return matrix, mask, {"thread": threading.get_ident(), "start": started, "end": time.perf_counter()}

    def __call__(self, bank, model, keypoints, descriptors, neighbors, index, geometric=True):
        if self.closed:
            raise RuntimeError("A closed owned geometry scheduler cannot execute")
        compatible = (cv2.__version__ == "4.13.0" and type(geometric) is bool
            and isinstance(keypoints, np.ndarray) and isinstance(descriptors, np.ndarray)
            and keypoints.ndim == descriptors.ndim == 2 and keypoints.shape[1] >= 2
            and keypoints.dtype in (np.dtype("float32"), np.dtype("float64"))
            and descriptors.dtype == np.float32 and descriptors.shape[1] == 128
            and len(keypoints) == len(descriptors) and len(descriptors) > 0
            and isinstance(neighbors, (list, tuple)) and 1 <= len(neighbors) <= self.maximum_neighbors
            and len(descriptors)*len(neighbors)*32 <= self.maximum_array_bytes
            and np.isfinite(keypoints[:, :2]).all() and np.isfinite(descriptors).all())
        self.trace["routes"].append({"route": "prepared_parallel" if compatible and self.parallel else "prepared_serial" if compatible else "original_before_execution",
            "query_rows": len(descriptors), "neighbors": len(neighbors)})
        if not compatible:
            return self.native(bank, model, keypoints, descriptors, neighbors, index, geometric)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        rows, jobs = [], []
        for image_id in neighbors:
            name, train_keypoints, train_descriptors = bank.original[image_id]
            pixels, point_ids, valid = index.get(model, image_id)
            if len(train_keypoints) != len(pixels):
                raise ValueError("Original keypoint and map observation identity mismatch")
            if not np.allclose(train_keypoints[:, :2], pixels, atol=1e-3):
                raise ValueError("Original SIFT pixels and map pixels disagree")
            nearest = matcher.knnMatch(descriptors, train_descriptors, k=2)
            reverse = matcher.match(train_descriptors, descriptors)
            reverse_map = {match.queryIdx: match.trainIdx for match in reverse}
            pairs = [match for pair in nearest if len(pair) == 2 for match, second in [pair]
                if match.distance < .8*second.distance and match.distance <= .7
                and reverse_map.get(match.trainIdx) == match.queryIdx]
            row = {"image_id": image_id, "name": name, "point_ids": point_ids, "valid": valid,
                "pairs": pairs, "raw_count": len(pairs), "matrix": None}
            rows.append(row)
            if geometric and len(pairs) >= 15:
                xy1 = np.asarray([keypoints[match.queryIdx, :2] for match in pairs])
                xy2 = np.asarray([train_keypoints[match.trainIdx, :2] for match in pairs])
                jobs.append((len(rows)-1, xy1, xy2))
        if sum(xy1.nbytes+xy2.nbytes for _, xy1, xy2 in jobs) > self.maximum_array_bytes:
            raise MemoryError("The preadmitted original geometry arrays exceeded their explicit budget")
        futures = []
        try:
            if self.parallel and len(jobs) > 1:
                if self.pool is None:
                    self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="owned-neighbor-F")
                for position, xy1, xy2 in jobs[:-1]:
                    futures.append((position, self.pool.submit(self.fit, xy1, xy2)))
                self.trace["peak_pending"] = max(self.trace["peak_pending"], len(futures))
                position, xy1, xy2 = jobs[-1]
                completed = [(position, self.fit(xy1, xy2))]
                completed.extend((position, future.result()) for position, future in futures)
            else:
                completed = [(position, self.fit(xy1, xy2)) for position, xy1, xy2 in jobs]
            for position, (matrix, mask, measurement) in completed:
                row = rows[position]
                row["matrix"] = matrix
                row["pairs"] = [] if matrix is None or mask is None else [match for match, ok in zip(row["pairs"], mask.ravel()) if ok]
                self.trace["geometry_jobs"].append({"image_id": row["image_id"], **measurement})
        except BaseException:
            for _, future in futures:
                future.cancel()
            raise
        correspondences, edges, all_pairs = [], [], []
        for row in rows:
            pairs, matrix = row["pairs"], row["matrix"]
            if geometric and len(pairs) < 15:
                pairs = []
            edges.append({"train_image_id": row["image_id"], "train_image_name": row["name"],
                "mutual_ratio_matches": row["raw_count"], "geometric_inliers": len(pairs),
                "F_from_query_localization_features": None if matrix is None else matrix.tolist()})
            for match in pairs:
                all_pairs.append((row["image_id"], match.queryIdx, match.trainIdx, match.distance))
                if row["valid"][match.trainIdx]:
                    correspondences.append((match.distance, match.queryIdx, int(row["point_ids"][match.trainIdx]),
                        row["image_id"], match.trainIdx))
        unique, used_query, used_point = [], set(), set()
        for distance, query, point_id, image_id, train_index in sorted(correspondences):
            if query in used_query or point_id in used_point:
                continue
            used_query.add(query)
            used_point.add(point_id)
            unique.append({"query": query, "point_id": point_id, "train_image_id": image_id,
                "train_index": train_index, "distance": float(distance)})
        return unique, edges, all_pairs

    def close(self):
        if not self.closed:
            started = time.perf_counter()
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.closed = True
            self.trace.update(closed=True, shutdown_seconds=time.perf_counter()-started)


@contextmanager
def parallel_neighbor_geometry(mode):
    import native_observation_index
    original = native_observation_index.indexed_match
    if mode not in ("incumbent", "serial", "parallel"):
        raise ValueError("An unsupported owned geometry route was selected")
    scheduler = None if mode == "incumbent" else ParallelNeighborGeometry(original, mode == "parallel")
    if scheduler is not None:
        native_observation_index.indexed_match = scheduler
    try:
        yield scheduler.trace if scheduler else {"mode": "incumbent", "setup_seconds": 0., "shutdown_seconds": 0.}
    finally:
        native_observation_index.indexed_match = original
        if scheduler is not None:
            scheduler.close()
