"""Compute frozen CPU SIFT queries in two bounded owned processes."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import multiprocessing
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import psutil
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"),
    str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002")]
from query_feature_cache import file_identity, option_identity
import feature_localization

_WORKER = None
_WORKER_IDENTITY = None


def options():
    selected = pycolmap.FeatureExtractionOptions()
    selected.sift.max_num_features, selected.sift.peak_threshold = 12000, .004
    return selected


def initialize_worker():
    global _WORKER, _WORKER_IDENTITY
    cv2.setNumThreads(8)
    _WORKER = pycolmap.Sift(options(), pycolmap.Device.cpu)
    process = psutil.Process()
    _WORKER_IDENTITY = {"pid": process.pid, "creation_time": process.create_time()}


def extract_worker(path):
    started = time.perf_counter()
    process = psutil.Process()
    before_cpu = process.cpu_times()
    path = Path(path)
    before = file_identity(path)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Image decode failed: {path}")
    keypoints, descriptors = _WORKER.extract(gray)
    after = file_identity(path)
    if before != after:
        raise ValueError("An owned SIFT source changed during extraction")
    after_cpu = process.cpu_times()
    return {"keypoints": keypoints, "descriptors": descriptors, "source": before,
        "gray_sha256": hashlib.sha256(gray.tobytes()).hexdigest(), "gray_shape": tuple(gray.shape),
        "options": option_identity(_WORKER.options.todict()), "device": _WORKER.device.name,
        "worker": _WORKER_IDENTITY, "whole_task_seconds": time.perf_counter()-started,
        "CPU_seconds": after_cpu.user+after_cpu.system-before_cpu.user-before_cpu.system}


class NativeSiftProcess:
    def __init__(self, paths):
        self.paths = tuple(Path(path).resolve() for path in paths)
        if not self.paths or len(self.paths) > 256:
            raise ValueError("A bounded cold ordered query list is required")
        self.pool, self.native = None, None
        self.pending = deque()
        self.next_submit, self.next_deliver = 0, 0
        self.closed = False
        self.trace = {"workers": 2, "capacity": 2, "submitted": 0, "delivered": 0,
            "peak_pending": 0, "setup_seconds": 0., "close_seconds": 0., "parent_verification_seconds": 0.,
            "worker_results": [], "saved_workers": [], "closed": False,
            "original_cache_and_feature_bank_producer_unchanged": True}

    def bind(self, native):
        if self.native is not None or type(native) is not pycolmap.Sift:
            raise ValueError("Exactly one native CPU extractor owns an ordered process broker")
        if (native.device != pycolmap.Device.cpu or option_identity(native.options.todict()) != option_identity(options().todict())):
            raise ValueError("The original CPU extractor options changed before process startup")
        self.native = native
        return SiftProxy(self)

    def fill(self):
        while len(self.pending) < 2 and self.next_submit < len(self.paths):
            path = self.paths[self.next_submit]
            self.pending.append((path, self.pool.submit(extract_worker, str(path))))
            self.next_submit += 1
            self.trace["submitted"] += 1
            self.trace["peak_pending"] = max(self.trace["peak_pending"], len(self.pending))

    def extract(self, gray):
        if self.closed or self.native is None or self.next_deliver >= len(self.paths):
            raise RuntimeError("The owned query order, extractor binding or lifecycle changed")
        if self.pool is None:
            started = time.perf_counter()
            self.pool = ProcessPoolExecutor(max_workers=2, initializer=initialize_worker,
                mp_context=multiprocessing.get_context("spawn"))
            self.fill()
            self.trace["setup_seconds"] += time.perf_counter()-started
        path, future = self.pending.popleft()
        value = future.result()
        started = time.perf_counter()
        if (value["source"] != file_identity(path) or value["gray_shape"] != tuple(gray.shape)
                or value["gray_sha256"] != hashlib.sha256(gray.tobytes()).hexdigest()
                or value["options"] != option_identity(self.native.options.todict())
                or value["device"] != self.native.device.name):
            raise ValueError("The caller query pixels, source or native SIFT options differ from the owned result")
        self.trace["parent_verification_seconds"] += time.perf_counter()-started
        keypoints, descriptors = value.pop("keypoints"), value.pop("descriptors")
        self.trace["worker_results"].append({"ordinal": self.next_deliver, "path": str(path), **value})
        self.next_deliver += 1
        self.trace["delivered"] += 1
        self.fill()
        return keypoints, descriptors

    def close(self):
        if self.closed:
            return
        started = time.perf_counter()
        if self.pool is not None:
            processes = list(self.pool._processes.values())
            for process in processes:
                try:
                    self.trace["saved_workers"].append({"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()})
                except psutil.NoSuchProcess:
                    pass
            for _, future in self.pending:
                future.cancel()
            self.pool.shutdown(wait=True, cancel_futures=True)
            for identity in self.trace["saved_workers"]:
                try:
                    process = psutil.Process(identity["pid"])
                    if process.create_time() == identity["creation_time"] and process.status() != psutil.STATUS_ZOMBIE:
                        raise RuntimeError("An owned SIFT process survived its complete shutdown")
                except psutil.NoSuchProcess:
                    pass
        self.pending.clear()
        self.closed = True
        self.trace.update(closed=True, close_seconds=time.perf_counter()-started)


class SiftProxy:
    def __init__(self, owner):
        self.owner = owner

    @property
    def options(self):
        return self.owner.native.options

    @property
    def device(self):
        return self.owner.native.device

    def extract(self, gray):
        return self.owner.extract(gray)


@contextmanager
def native_sift_process(paths, enabled=True):
    started = time.perf_counter()
    bank = feature_localization.FeatureBank
    prior_own = bank.__dict__.get("__setattr__")
    original = bank.__setattr__
    broker = NativeSiftProcess(paths) if enabled else None
    trace = {"broker": broker.trace if broker else None, "setup_seconds": 0., "restore_seconds": 0.,
        "binding_restored": False, "only_native_extract_moves_original_extra_and_cache_unchanged": True}
    def assign(self, name, value):
        if name == "extractor":
            value = broker.bind(value)
        original(self, name, value)
    try:
        if broker is not None:
            bank.__setattr__ = assign
        trace["setup_seconds"] = time.perf_counter()-started
        yield trace
    finally:
        restored = time.perf_counter()
        if broker is not None:
            if prior_own is None:
                del bank.__setattr__
            else:
                bank.__setattr__ = prior_own
            broker.close()
        trace.update(binding_restored=bank.__dict__.get("__setattr__") is prior_own,
            restore_seconds=time.perf_counter()-restored)
