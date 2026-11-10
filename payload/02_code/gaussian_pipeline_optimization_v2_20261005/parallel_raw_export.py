"""Prepare raw sensor JPEG exports in a bounded ordered CPU queue."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import time

import cv2
import numpy as np


def prepared_exports(messages, decoder, converter, quality, workers=4, capacity=8, encode_in_workers=True, trace=None):
    if workers not in (1, 4) or capacity != 8 or type(encode_in_workers) is not bool:
        raise ValueError("Raw export preparation uses only its explicit bounded CPU configuration")
    trace = {} if trace is None else trace
    trace.update(workers=workers, capacity=capacity, encode_in_workers=encode_in_workers,
        maximum_pending=0, submitted=0, consumed=0, read_seconds=0., wait_seconds=0.,
        preparation_seconds_sum=0., maximum_prepared_image_bytes=0, closed=False)
    source = iter(messages)
    encoder = cv2.imencode
    parameters = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    jobs = deque()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="raw-export") if workers > 1 else None
    exhausted = False

    def prepare(frame, payload):
        started = time.perf_counter()
        stamp, _, raw = decoder(payload)
        image = converter(raw)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.nbytes > 16*1024**2:
            raise ValueError("A raw export worker received an unsupported BGR image")
        if encode_in_workers:
            ok, encoded = encoder(".jpg", image, parameters)
            if not ok:
                raise RuntimeError("A raw export worker could not encode the original JPEG")
            result = (int(frame), int(stamp), ok, encoded)
        else:
            result = (int(frame), int(stamp), image)
        return result, time.perf_counter()-started, image.nbytes

    def pull():
        nonlocal exhausted
        while not exhausted and len(jobs) < capacity:
            started = time.perf_counter()
            try:
                frame, message = next(source)
            except StopIteration:
                exhausted = True
                break
            finally:
                trace["read_seconds"] += time.perf_counter()-started
            payload = message["payload"]
            if not isinstance(payload, bytes):
                payload = bytes(payload)
            if len(payload) > 16*1024**2:
                raise MemoryError("A pending raw export payload exceeds its declared bound")
            jobs.append(pool.submit(prepare, frame, payload))
            trace["submitted"] += 1
            trace["maximum_pending"] = max(trace["maximum_pending"], len(jobs))

    try:
        if pool:
            pull()
            while jobs:
                started = time.perf_counter()
                result, seconds, image_bytes = jobs.popleft().result()
                trace["wait_seconds"] += time.perf_counter()-started
                trace["preparation_seconds_sum"] += seconds
                trace["maximum_prepared_image_bytes"] = max(trace["maximum_prepared_image_bytes"], image_bytes)
                if not encode_in_workers:
                    frame, stamp, image = result
                    ok, encoded = encoder(".jpg", image, parameters)
                    if not ok:
                        raise RuntimeError("The ordered main-thread JPEG cache could not encode an image")
                    result = (frame, stamp, ok, encoded)
                trace["consumed"] += 1
                yield result
                pull()
        else:
            for frame, message in source:
                payload = message["payload"]
                result, seconds, image_bytes = prepare(frame, payload)
                trace["preparation_seconds_sum"] += seconds
                trace["maximum_prepared_image_bytes"] = max(trace["maximum_prepared_image_bytes"], image_bytes)
                if not encode_in_workers:
                    frame, stamp, image = result
                    ok, encoded = encoder(".jpg", image, parameters)
                    if not ok:
                        raise RuntimeError("The serial main-thread JPEG cache could not encode an image")
                    result = (frame, stamp, ok, encoded)
                trace["submitted"] += 1
                trace["consumed"] += 1
                trace["maximum_pending"] = 1
                yield result
        if trace["submitted"] != trace["consumed"]:
            raise ValueError("The ordered export queue left an unconsumed selected frame")
    finally:
        for future in jobs:
            future.cancel()
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        if hasattr(source, "close"):
            source.close()
        trace["closed"] = True


@contextmanager
def parallel_export_preparation(module, enabled=True, encode_in_workers=True):
    original = module.main
    source = inspect.getsource(original)
    before = '''    for exported, (frame, message) in enumerate(image_messages(selected_by_id), 1):
        stamp, _, raw = decode_bayer(message["payload"])
        row = selected_by_id[frame]
        if int(stamp) != row["camera_header_ns"]:
            raise ValueError("Raw image identity changed before export")
        ok, encoded = cv2.imencode(".jpg", demosaic_rggb8_to_bgr(raw),
                                 [cv2.IMWRITE_JPEG_QUALITY, baseline.POLICY["jpeg_quality"]])
'''
    after = '''    for exported, (frame, stamp, ok, encoded) in enumerate(_prepared_raw_exports(
            image_messages(selected_by_id), decode_bayer, demosaic_rggb8_to_bgr, baseline.POLICY["jpeg_quality"]), 1):
        row = selected_by_id[frame]
        if int(stamp) != row["camera_header_ns"]:
            raise ValueError("Raw image identity changed before export")
'''
    if source.count(before) != 1:
        raise ValueError("The original ordered raw export block changed")
    records, generators = [], []

    def generate(messages, decoder, converter, quality):
        record = {}
        records.append(record)
        generator = prepared_exports(messages, decoder, converter, quality, workers=4 if enabled else 1,
            capacity=8, encode_in_workers=encode_in_workers, trace=record)
        generators.append(generator)
        return generator

    namespace = dict(module.__dict__)
    namespace["_prepared_raw_exports"] = generate
    exec(compile(source.replace(before, after), str(module.__file__), "exec"), namespace)
    module.main = namespace["main"]
    try:
        yield records
    finally:
        for generator in generators:
            generator.close()
        module.main = original
