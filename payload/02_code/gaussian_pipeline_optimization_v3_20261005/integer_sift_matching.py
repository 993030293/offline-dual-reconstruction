"""Adapt only native BF distances while retaining all geometry and pose code."""

from contextlib import contextmanager
import json
from pathlib import Path
import struct
import subprocess
import time

import cv2
import numpy as np

HEADER = struct.Struct("<I")


def bytes_exact(values):
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 128 or not np.isfinite(values).all():
        raise ValueError("The original finite float32 native query contract changed")
    scaled = values*512
    if not (scaled == np.rint(scaled)).all() or scaled.min() < 0 or scaled.max() > 255:
        raise ValueError("A native SIFT component cannot be represented exactly as a byte")
    result = scaled.astype(np.uint8)
    if not np.array_equal(result.astype(np.float32)/512, values):
        raise ValueError("The original float32 SIFT representation did not recover exactly")
    return result


def read_exact(stream, count):
    output = bytearray()
    while len(output) < count:
        block = stream.read(count-len(output))
        if not block:
            raise EOFError("The owned exact SIFT worker terminated before its complete result")
        output.extend(block)
    return output


class DistanceWorker:
    def __init__(self, log):
        source = Path(__file__).with_name("integer_sift_worker.py").resolve().as_posix()
        if source[1:3] != ":/" or source[0].lower() != "c":
            raise ValueError("The owned worker source must stay in the existing C drive project")
        source = "/mnt/c/"+source[3:]
        self.command = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
            "/opt/e26_thin_surface_v1/venv/bin/python", source]
        self.log = Path(log).open("xb")
        started = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.stopped = False
        self.receipt = {"command": self.command, "pid": self.process.pid, "calls": [], "counted": False}
        self.receipt["info"] = self.rpc({"command": "info"})
        self.receipt["info"].pop("payload")
        self.receipt["startup_seconds"] = time.perf_counter()-started

    def rpc(self, record, payload=b""):
        encoded = json.dumps(record, separators=(",", ":")).encode()
        for block in (HEADER.pack(len(encoded)), encoded, payload):
            view = memoryview(block)
            while view:
                written = self.process.stdin.write(view)
                if not written:
                    raise EOFError("The owned matching request pipe stopped accepting bytes")
                view = view[written:]
        self.process.stdin.flush()
        size = HEADER.unpack(read_exact(self.process.stdout, HEADER.size))[0]
        if not 0 < size <= 4096:
            raise ValueError("The matching response metadata exceeds its strict bound")
        reply = json.loads(read_exact(self.process.stdout, size))
        if reply["status"] == "failed":
            raise RuntimeError(reply["error"])
        count = reply["payload_bytes"]
        if not isinstance(count, int) or not 0 <= count <= 12*12000*20:
            raise ValueError("The matching response allocation exceeds its strict bound")
        reply["payload"] = read_exact(self.process.stdout, count)
        return reply

    def match(self, query, train):
        started = time.perf_counter()
        q, t = bytes_exact(query), bytes_exact(train)
        reply = self.rpc({"command": "nearest", "query_rows": len(q), "train_rows": len(t)}, q.tobytes()+t.tobytes())
        if reply["status"] != "matched" or reply["query_rows"] != len(q) or reply["train_rows"] != len(t) or len(reply["payload"]) != len(q)*16+len(t)*4:
            raise ValueError("The actual integer matcher response identity changed")
        payload = reply.pop("payload")
        forward = np.frombuffer(payload, np.int32, count=len(q)*4).reshape(len(q), 4).copy()
        # CPU sqrt preserves the same native float32 distance rounding.
        distances = np.sqrt(forward[:, (1, 3)].astype(np.float32)/262144)
        reverse = np.frombuffer(payload, np.int32, count=len(t), offset=len(q)*16).copy()
        self.receipt["calls"].append(dict(reply, total_prepare_IPC_kernel_seconds=time.perf_counter()-started))
        return forward, distances, reverse

    def batch(self, query, trains):
        started = time.perf_counter()
        q, values = bytes_exact(query), [bytes_exact(t) for t in trains]
        counts = [len(t) for t in values]
        reply = self.rpc({"command": "batch", "query_rows": len(q), "train_counts": counts},
            q.tobytes()+b"".join(t.tobytes() for t in values))
        payload = reply.pop("payload")
        expected = len(q)*16*len(trains)+sum(counts)*4
        if reply["status"] != "batched" or reply["query_rows"] != len(q) or reply["train_counts"] != counts or len(payload) != expected:
            raise ValueError("The complete native SIFT batch response identity changed")
        result, offset = [], 0
        for count in counts:
            forward = np.frombuffer(payload, np.int32, count=len(q)*4, offset=offset).reshape(len(q), 4).copy()
            offset += len(q)*16
            reverse = np.frombuffer(payload, np.int32, count=count, offset=offset).copy()
            offset += count*4
            distance = np.sqrt(forward[:, (1, 3)].astype(np.float32)/262144)
            result.append((forward, distance, reverse))
        self.receipt["calls"].append(dict(reply, total_prepare_IPC_kernel_seconds=time.perf_counter()-started))
        return result

    def close(self):
        if not self.stopped:
            started = time.perf_counter()
            try:
                receipt = self.rpc({"command": "stop"})
                receipt.pop("payload")
                self.receipt["shutdown"] = receipt
            finally:
                self.process.stdin.close()
                self.process.stdout.close()
                self.receipt["returncode"] = self.process.wait(timeout=30)
                self.log.close()
                self.stopped = True
                self.receipt["shutdown_seconds"] = time.perf_counter()-started
            if self.receipt["returncode"] != 0:
                raise RuntimeError("The owned exact SIFT worker did not shut down successfully")


class ExactMatcher:
    def __init__(self, worker, prepared):
        self.worker, self.pending, self.prepared = worker, None, list(prepared)

    def knnMatch(self, query, train, k):
        if k != 2 or self.pending is not None:
            raise ValueError("The exact matcher only implements the original forward2/reverse1 contract")
        if not self.prepared:
            raise ValueError("A native neighbor consumed more exact batches than registered")
        expected_query, expected_train, values = self.prepared.pop(0)
        if not np.array_equal(query, expected_query) or not np.array_equal(train, expected_train):
            raise ValueError("The exact neighbor batch order or native descriptor values changed")
        forward, distances, reverse = values
        d = distances.astype(np.float64)
        mask = (d[:, 0] < .8*d[:, 1]) & (d[:, 0] <= .7) & (reverse[forward[:, 0]] == np.arange(len(query)))
        selected = np.flatnonzero(mask)
        self.pending = (query, train, [(int(forward[i, 0]), int(i)) for i in selected])
        return [[cv2.DMatch(int(i), int(forward[i, 0]), float(d[i, 0])),
            cv2.DMatch(int(i), int(forward[i, 2]), float(d[i, 1]))] for i in selected]

    def match(self, train, query):
        if self.pending is None or self.pending[0] is not query or self.pending[1] is not train:
            raise ValueError("The original paired reverse descriptor identity changed")
        reverse = self.pending[2]
        self.pending = None
        return [cv2.DMatch(i, index, 0.) for i, index in reverse]


@contextmanager
def integer_sift_matching(log, enabled=True):
    import feature_localization
    original = cv2.BFMatcher
    original_match = feature_localization.FeatureBank.match
    worker = None
    trace = {"active": enabled, "calls": [], "counted": False}
    batches = []
    try:
        if enabled:
            worker = DistanceWorker(log)

            def create(norm):
                if norm != cv2.NORM_L2 or len(batches) != 1:
                    raise ValueError("The exact matcher requires the original L2 norm")
                return ExactMatcher(worker, batches[0])

            def match(bank, model, keypoints, descriptors, neighbors, geometric=True):
                if batches:
                    raise ValueError("The owned exact native SIFT matcher cannot be nested")
                trains = [bank.original[i][2] for i in neighbors]
                results = worker.batch(descriptors, trains)
                batches.append([(descriptors, t, r) for t, r in zip(trains, results)])
                try:
                    return original_match(bank, model, keypoints, descriptors, neighbors, geometric)
                finally:
                    batches.pop()

            cv2.BFMatcher = create
            feature_localization.FeatureBank.match = match
        yield trace
    finally:
        cv2.BFMatcher = original
        feature_localization.FeatureBank.match = original_match
        if worker is not None:
            worker.close()
            trace.update(worker.receipt)
