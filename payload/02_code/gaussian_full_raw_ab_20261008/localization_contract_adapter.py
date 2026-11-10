"""Adapt descriptor storage and worker options without changing native algorithms."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
import time

import cv2
import numpy as np
import psutil
import pycolmap

import feature_localization
import native_sift_process as native_process
from byte_sift_matching import TransientFloatView
from native_observation_index import admission
from query_feature_cache import option_identity


@contextmanager
def exact_transient_training_view():
    """Expose bytes as original float32/512 values at the composed matcher boundary."""
    original = feature_localization.FeatureBank.match
    trace = {"actual_match_calls": 0, "transient_conversion_calls": 0,
        "maximum_transient_descriptor_bytes": 0, "maximum_observed_store_plus_transient_bytes": 0,
        "query_arrays_unchanged": True, "native_distance_scale_unchanged": 512,
        "map_admission_required": True, "binding_restored": False}

    def match(bank, model, keypoints, descriptors, neighbors, geometric=True):
        if (descriptors.dtype != np.float32 or descriptors.ndim != 2 or descriptors.shape[1] != 128
                or not np.isfinite(descriptors).all()):
            raise ValueError("The original finite normalized query descriptor contract changed")
        compatible, reason = admission(model)
        if not compatible:
            raise ValueError("This composition adapter requires the unchanged native map admission: " + reason)
        store = bank.original
        if isinstance(store, TransientFloatView):
            raise ValueError("A composed native descriptor view cannot be nested")
        bank.original = TransientFloatView(store, trace)
        trace["actual_match_calls"] += 1
        try:
            return original(bank, model, keypoints, descriptors, neighbors, geometric)
        finally:
            bank.original = store

    feature_localization.FeatureBank.match = match
    try:
        yield trace
    finally:
        feature_localization.FeatureBank.match = original
        trace["binding_restored"] = feature_localization.FeatureBank.match is original


def initialize_frozen_worker(frozen_options):
    """Construct a CPU extractor from the caller's complete, identical option dictionary."""
    options = pycolmap.FeatureExtractionOptions(frozen_options)
    if option_identity(options.todict()) != option_identity(frozen_options):
        raise ValueError("Worker option reconstruction changed a native extraction parameter")
    cv2.setNumThreads(options.num_threads)
    native_process._WORKER = pycolmap.Sift(options, pycolmap.Device.cpu)
    process = psutil.Process()
    native_process._WORKER_IDENTITY = {"pid": process.pid, "creation_time": process.create_time()}
    if option_identity(native_process._WORKER.options.todict()) != option_identity(frozen_options):
        raise ValueError("The worker SIFT constructor changed frozen extraction options")


class CompatibleNativeSiftProcess(native_process.NativeSiftProcess):
    """Retain the bounded broker and exact pixel/result checks with current caller threads."""

    def bind(self, native):
        if self.native is not None or type(native) is not pycolmap.Sift or native.device != pycolmap.Device.cpu:
            raise ValueError("Exactly one native CPU extractor must own the broker")
        threads = native.options.num_threads
        if not isinstance(threads, int) or not 0 < threads <= 8:
            raise ValueError("Two SIFT workers must stay within the declared 16-thread budget")
        expected = native_process.options()
        expected.num_threads = threads
        if option_identity(native.options.todict()) != option_identity(expected.todict()):
            raise ValueError("A native SIFT algorithm option changed beyond the caller thread configuration")
        self.frozen_options = native.options.todict()
        self.native = native
        self.trace.update(frozen_complete_options=self.frozen_options,
            worker_extraction_threads=threads, maximum_extraction_threads=2*threads,
            caller_and_worker_complete_options_equal=True)
        return native_process.SiftProxy(self)

    def extract(self, gray):
        if self.closed or self.native is None:
            raise RuntimeError("The owned broker lifecycle changed before worker startup")
        if self.pool is None:
            started = time.perf_counter()
            self.pool = ProcessPoolExecutor(max_workers=2, initializer=initialize_frozen_worker,
                initargs=(self.frozen_options,), mp_context=multiprocessing.get_context("spawn"))
            self.fill()
            self.trace["setup_seconds"] += time.perf_counter()-started
        return super().extract(gray)


@contextmanager
def exact_worker_options():
    """Swap only the private broker constructor and restore it on every exit path."""
    original = native_process.NativeSiftProcess
    native_process.NativeSiftProcess = CompatibleNativeSiftProcess
    trace = {"complete_options_forwarded": True, "algorithm_thresholds_unchanged": True,
        "binding_restored": False}
    try:
        yield trace
    finally:
        native_process.NativeSiftProcess = original
        trace["binding_restored"] = native_process.NativeSiftProcess is original
