"""Store native SIFT bytes and restore only the current neighbor's original float values."""

from contextlib import contextmanager

import numpy as np


class TransientFloatView:
    """Expose the original matcher representation without retaining float descriptors."""

    def __init__(self, store, trace):
        self.store, self.trace = store, trace

    def __getitem__(self, image_id):
        name, keypoints, descriptors = self.store[image_id]
        if descriptors.dtype != np.uint8 or descriptors.ndim != 2 or descriptors.shape[1] != 128:
            raise ValueError("The compressed training store is not a native128-component byte SIFT record")
        values = descriptors.astype(np.float32)
        values /= 512.
        self.trace["transient_conversion_calls"] += 1
        self.trace["maximum_transient_descriptor_bytes"] = max(self.trace["maximum_transient_descriptor_bytes"], values.nbytes)
        if hasattr(self.store, "receipt"):
            storage = self.store.receipt()["cached_array_bytes"]
            self.trace["maximum_observed_store_plus_transient_bytes"] = max(
                self.trace["maximum_observed_store_plus_transient_bytes"], storage+values.nbytes)
        return name, keypoints, values


@contextmanager
def byte_sift_matching(enabled=True):
    import feature_localization
    import sift_feature_store

    original_decode = sift_feature_store.SiftFeatureStore.decode
    original_match = feature_localization.FeatureBank.match
    trace = {"decode_calls": 0, "actual_match_calls": 0, "transient_conversion_calls": 0,
        "maximum_transient_descriptor_bytes": 0, "maximum_observed_store_plus_transient_bytes": 0,
        "original_float32_native_matcher_unchanged": True, "query_descriptors_unchanged": True}

    def decode(store, image_id, record):
        name, kr, kc, kd, dr, dc, dd = record
        if name != store.names[image_id] or kr != dr or dc != 128:
            raise ValueError("Byte SIFT row identity or descriptor shape changed")
        keypoints = np.frombuffer(kd, np.float32).reshape(kr, kc).copy()
        descriptors = np.frombuffer(dd, np.uint8).reshape(dr, dc).copy()
        trace["decode_calls"] += 1
        return name, keypoints, descriptors

    def match(bank, model, keypoints, descriptors, neighbors, geometric=True):
        if descriptors.dtype != np.float32 or descriptors.ndim != 2 or descriptors.shape[1] != 128 or not np.isfinite(descriptors).all():
            raise ValueError("The original float32 query descriptor contract changed")
        store = bank.original
        if isinstance(store, TransientFloatView):
            raise ValueError("A transient feature view cannot be nested")
        trace["actual_match_calls"] += 1
        bank.original = TransientFloatView(store, trace)
        try:
            return original_match(bank, model, keypoints, descriptors, neighbors, geometric)
        finally:
            bank.original = store

    if enabled:
        sift_feature_store.SiftFeatureStore.decode = decode
        feature_localization.FeatureBank.match = match
    try:
        yield trace
    finally:
        sift_feature_store.SiftFeatureStore.decode = original_decode
        feature_localization.FeatureBank.match = original_match
