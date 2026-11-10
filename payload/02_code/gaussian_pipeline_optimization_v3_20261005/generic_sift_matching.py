"""Select a documented exact matcher before execution, preserving native geometry."""

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from integer_sift_matching import DistanceWorker, ExactMatcher, bytes_exact

INPUT_BYTES = 32*1024**2
REPLY_BYTES = 12*12000*20
MATRIX_ELEMENTS = 144_000_000


def admission(query, trains):
    arrays = [query, *trains]
    if not trains or len(trains) > 12 or len(query) < 1 or any(len(t) < 2 for t in trains):
        return "native", "insufficient_support_or_neighbor_count"
    if any(a.dtype != np.float32 or a.ndim != 2 or a.shape[1] != 128 or not np.isfinite(a).all() for a in arrays):
        return "native", "original_BF_input_format"
    if (len(query)+sum(map(len, trains)))*128 > INPUT_BYTES or len(query)*16*len(trains)+sum(map(len, trains))*4 > REPLY_BYTES:
        return "native", "transport_budget"
    pq = max(32, (len(query)+15)//16*16)
    if any(pq*max(8, (len(t)+7)//8*8) > MATRIX_ELEMENTS for t in trains):
        return "native", "matrix_budget"
    for value in arrays:
        try:
            bytes_exact(value)
        except ValueError:
            return "native", "descriptor_not_exactly_byte_representable"
    return "integer", "all_original_numerical_and_resource_bounds_met"


class GenericWorker(DistanceWorker):
    def __init__(self, log):
        source = Path(__file__).with_name("generic_sift_worker.py").resolve().as_posix()
        if source[1:3] != ":/" or source[0].lower() != "c":
            raise ValueError("The worker must stay inside the existing C drive project")
        self.command = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
            "/opt/e26_thin_surface_v1/venv/bin/python", "/mnt/c/"+source[3:]]
        self.log = Path(log).open("xb")
        started = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.stopped = False
        self.receipt = {"command": self.command, "pid": self.process.pid, "calls": [], "counted": False}
        try:
            self.receipt["info"] = self.rpc({"command": "info"})
            self.receipt["info"].pop("payload")
        except BaseException:
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.wait(timeout=30)
            self.log.close()
            raise
        self.receipt["startup_seconds"] = time.perf_counter()-started


@contextmanager
def generic_sift_matching(log, enabled=True):
    import feature_localization
    original_factory, original_match = cv2.BFMatcher, feature_localization.FeatureBank.match
    batches, worker = [], None
    trace = {"active": enabled, "routes": [], "counted": False, "new_optimization_count": 0}
    try:
        if enabled:
            def factory(norm):
                if norm != cv2.NORM_L2:
                    raise ValueError("The original native L2 matcher contract changed")
                return ExactMatcher(worker, batches[0]) if batches else original_factory(norm)

            def match(bank, model, keypoints, descriptors, neighbors, geometric=True):
                nonlocal worker
                if batches:
                    raise ValueError("The owned matching context cannot be nested")
                trains = [bank.original[i][2] for i in neighbors]
                route, reason = admission(descriptors, trains)
                trace["routes"].append({"route": route, "reason": reason, "query_rows": len(descriptors),
                    "train_rows": list(map(len, trains)), "all_features_preserved": True})
                if route == "native":
                    return original_match(bank, model, keypoints, descriptors, neighbors, geometric)
                if worker is None:
                    worker = GenericWorker(log)
                values = worker.batch(descriptors, trains)
                batches.append([(descriptors, t, r) for t, r in zip(trains, values)])
                try:
                    return original_match(bank, model, keypoints, descriptors, neighbors, geometric)
                finally:
                    batches.pop()

            cv2.BFMatcher, feature_localization.FeatureBank.match = factory, match
        yield trace
    finally:
        active_error = sys.exception()
        cv2.BFMatcher, feature_localization.FeatureBank.match = original_factory, original_match
        if worker is not None:
            try:
                worker.close()
            except BaseException as cleanup_error:
                trace["cleanup_error"] = repr(cleanup_error)
                if active_error is None:
                    raise
                active_error.add_note("Owned worker cleanup also failed: "+repr(cleanup_error))
            finally:
                trace["worker"] = worker.receipt

