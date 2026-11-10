"""Bounded raw-image scoring with equivalent integer derivatives and keyed reuse."""

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import os
import tempfile
import time

import cv2
import numpy as np


FIELDS = ("sharpness", "clipped_fraction", "mean_brightness",
          "corner_count_half_resolution", "occupied_grid_cells_4x4", "quality_score")


def compact_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The default 3x3 cross derivative of uint8 is exactly representable in int16.
    derivative = cv2.Laplacian(gray, cv2.CV_16S)
    _, deviation = cv2.meanStdDev(derivative)
    sharpness = float(deviation[0, 0] ** 2)
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    reduced = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    corners = cv2.goodFeaturesToTrack(reduced, maxCorners=800, qualityLevel=.01,
                                      minDistance=5, blockSize=5)
    count, occupied = 0, 0
    if corners is not None:
        count = len(corners)
        cells = np.minimum(3, np.floor(corners[:, 0, :] /
                           [reduced.shape[1]/4, reduced.shape[0]/4]).astype(int))
        occupied = len(set(map(tuple, cells)))
    score = math.log1p(sharpness)+.25*math.log1p(count)+.4*occupied/16-4*clipped
    return dict(zip(FIELDS, [sharpness, clipped, float(gray.mean()), count, occupied, score]))


class ScoreCache:
    """Cache image-only scores; pose-dependent decisions always remain downstream."""

    def __init__(self, directory, contract):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.version = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    def key(self, payload):
        digest = hashlib.sha256(self.version.encode())
        digest.update(payload)
        return digest.hexdigest()

    def read(self, key):
        path = self.directory/(key+".json")
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        scores = record.get("quality", {})
        if record.get("key") != key or tuple(sorted(scores)) != tuple(sorted(FIELDS)):
            raise ValueError(f"Invalid quality cache identity/schema: {path}")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scores.values()):
            raise ValueError(f"Invalid quality cache value: {path}")
        canonical = json.dumps(scores, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if record.get("quality_sha256") != hashlib.sha256(canonical).hexdigest():
            raise ValueError(f"Invalid quality cache payload checksum: {path}")
        return scores

    def write(self, key, scores):
        path = self.directory/(key+".json")
        canonical = json.dumps(scores, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        record = {"key": key, "quality": scores, "quality_sha256": hashlib.sha256(canonical).hexdigest()}
        # Unique temporary names permit idempotent producers without exposing partial JSON.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                         prefix=key+".", suffix=".tmp", delete=False) as stream:
            json.dump(record, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.replace(path)


def score_messages(messages, decode, demosaic, scorer, workers=1, capacity=16, cache=None,
                   maximum_seconds=300, progress=None):
    """Use one indexed reader and bounded workers; emit original chronological order."""
    if workers < 1 or capacity < workers:
        raise ValueError("Queue capacity must cover all workers")
    io_seconds, submitted, peak_pending = 0., 0, 0
    started = time.perf_counter()

    def process(frame, message):
        payload = message["payload"]
        before = time.perf_counter()
        stamp, optical, raw = decode(payload)
        header_seconds = time.perf_counter()-before
        metadata = {"camera_frame": frame, "camera_header_ns": int(stamp), "optical_frame": optical,
                    "width": int(raw.shape[1]), "height": int(raw.shape[0])}
        before = time.perf_counter()
        key = cache.key(payload) if cache is not None else None
        cached = cache.read(key) if cache is not None else None
        cache_seconds = time.perf_counter()-before
        if cached is not None:
            return {**metadata, "quality": cached, "cache_hit": True,
                    "decode_seconds": header_seconds, "score_seconds": 0., "cache_seconds": cache_seconds}
        before = time.perf_counter()
        image = demosaic(raw)
        decoded = time.perf_counter()
        scores = scorer(image)
        scored = time.perf_counter()
        if cache is not None:
            cache.write(key, scores)
        return {**metadata, "quality": scores, "cache_hit": False, "decode_seconds": header_seconds+decoded-before,
                "score_seconds": scored-decoded,
                "cache_seconds": cache_seconds+time.perf_counter()-scored}

    iterator = iter(messages)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        exhausted = False
        while not exhausted or pending:
            while not exhausted and len(pending) < capacity:
                if time.perf_counter()-started > maximum_seconds:
                    raise TimeoutError("Raw scoring exceeded its registered time budget")
                before = time.perf_counter()
                try:
                    frame, message = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                finally:
                    io_seconds += time.perf_counter()-before
                submitted += 1
                if workers == 1:
                    rows.append(process(frame, message))
                    if progress is not None:
                        progress(len(rows))
                else:
                    pending.append(pool.submit(process, frame, message))
                    peak_pending = max(peak_pending, len(pending))
            if pending:
                rows.append(pending.popleft().result())
                if progress is not None:
                    progress(len(rows))
    return rows, {"read_seconds": io_seconds, "submitted": submitted,
                  "peak_pending": peak_pending, "queue_capacity": capacity,
                  "cache_hits": sum(row["cache_hit"] for row in rows),
                  "worker_decode_seconds_sum": sum(row["decode_seconds"] for row in rows),
                  "worker_score_seconds_sum": sum(row["score_seconds"] for row in rows),
                  "worker_cache_seconds_sum": sum(row["cache_seconds"] for row in rows)}
