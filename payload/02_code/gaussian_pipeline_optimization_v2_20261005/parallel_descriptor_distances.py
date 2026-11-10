"""Overlap independent descriptor distances while keeping geometry and RNG serial."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import textwrap
import time

import cv2
import numpy as np


def distance_pair(query, training):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    return matcher.knnMatch(query, training, k=2), matcher.match(training, query)


class NeighborDistanceQueue:
    """Materialize mutable lazy records on the calling thread before native work."""

    def __init__(self, store, query, neighbors, trace):
        self.store, self.query, self.neighbors, self.trace = store, query, list(neighbors), trace
        if len(set(self.neighbors)) != len(self.neighbors) or query.dtype != np.float32 or not np.isfinite(query).all():
            raise ValueError("Descriptor distance work requires unique neighbors and original finite float32 queries")
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="descriptor-distance")
        self.pending, self.next_index, self.request_index = {}, 0, 0

    def fill(self):
        while self.next_index < len(self.neighbors) and len(self.pending) < 2:
            image_id = self.neighbors[self.next_index]
            name, keypoints, descriptors = self.store[image_id]
            if descriptors.dtype != np.float32 or descriptors.ndim != 2 or descriptors.shape[1] != 128:
                raise ValueError("A queued training descriptor lost its native normalized float32 representation")
            future = self.pool.submit(distance_pair, self.query, descriptors)
            self.pending[image_id] = (name, keypoints, future)
            self.next_index += 1
            self.trace["submitted_neighbors"] += 1
            self.trace["maximum_pending"] = max(self.trace["maximum_pending"], len(self.pending))
            self.trace["maximum_float_descriptor_bytes_per_job"] = max(self.trace["maximum_float_descriptor_bytes_per_job"], descriptors.nbytes)

    def take(self, image_id):
        if self.request_index >= len(self.neighbors) or self.neighbors[self.request_index] != image_id:
            raise ValueError("Geometry requested a neighbor outside its original serial order")
        self.fill()
        name, keypoints, future = self.pending.pop(image_id)
        nearest, reverse = future.result()
        self.request_index += 1
        self.trace["consumed_neighbors"] += 1
        self.fill()
        return name, keypoints, nearest, reverse

    def close(self):
        for _, _, future in self.pending.values():
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()


@contextmanager
def parallel_descriptor_distances(enabled=True):
    import feature_localization
    original = feature_localization.FeatureBank.match
    original_source = inspect.getsource(original)
    source = textwrap.dedent(original_source)
    replacements = [
        ("def match(self, model, keypoints, descriptors, neighbors, geometric=True):",
         "def match(self, model, keypoints, descriptors, neighbors, geometric=True, distance_queue=None):"),
        ("        name, train_keypoints, train_descriptors = self.original[image_id]",
         "        name, train_keypoints, nearest, reverse = distance_queue.take(image_id)"),
        ("        nearest = matcher.knnMatch(descriptors, train_descriptors, k=2)\n        reverse = matcher.match(train_descriptors, descriptors)", "")]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("The original descriptor matcher no longer matches the guarded serial geometry adapter")
        source = source.replace(before, after)
    namespace = dict(original.__globals__)
    exec(compile(source, "<bounded_parallel_distances_serial_geometry>", "exec"), namespace)
    compiled = namespace["match"]
    trace = {"actual_calls": 0, "matching_seconds": 0., "submitted_neighbors": 0, "consumed_neighbors": 0,
        "maximum_pending": 0, "maximum_float_descriptor_bytes_per_job": 0,
        "workers": 2, "internal_OpenCV_threads": 4, "geometry_RNG_and_neighbor_order_serial": True,
        "query_features_unchanged": True, "mutable_lazy_store_access_on_calling_thread_only": True}

    def match(bank, model, keypoints, descriptors, neighbors, geometric=True):
        before = time.perf_counter()
        previous_threads = cv2.getNumThreads()
        queue = NeighborDistanceQueue(bank.original, descriptors, neighbors, trace)
        cv2.setNumThreads(4)
        try:
            result = compiled(bank, model, keypoints, descriptors, neighbors, geometric, queue)
            if queue.request_index != len(neighbors) or queue.pending:
                raise ValueError("A complete matching call did not consume its independent descriptor work")
            trace["actual_calls"] += 1
            return result
        finally:
            queue.close()
            cv2.setNumThreads(previous_threads)
            trace["matching_seconds"] += time.perf_counter()-before

    if enabled:
        feature_localization.FeatureBank.match = match
    try:
        yield trace
    finally:
        feature_localization.FeatureBank.match = original

