"""Retain native quality semantics while sharing one full gray histogram scan."""

from contextlib import contextmanager
import inspect
import time

import cv2
import numpy as np

PIXEL_LIMIT = 2**24
LEVELS = np.arange(256, dtype=np.uint64)


def gray_moments(gray):
    counts = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.uint64)
    count = gray.size
    if int(counts.sum(dtype=np.uint64)) != count:
        raise ValueError("The complete original gray histogram denominator changed")
    brightness = int(counts @ LEVELS)/count
    clipped = (int(counts[:6].sum(dtype=np.uint64))+int(counts[250:].sum(dtype=np.uint64)))/count
    return float(brightness), float(clipped)


def make_quality(original):
    started = time.perf_counter()
    source = inspect.getsource(original)
    before = "    clipped = float(np.mean((gray <= 5) | (gray >= 250)))\n"
    after = "float(gray.mean())"
    if source.count(before) != 1 or source.count(after) != 1:
        raise ValueError("The complete native quality moment anchors changed")
    namespace = dict(original.__globals__)
    namespace["_gray_moments"] = gray_moments
    modified = source.replace(before, "    brightness, clipped = _gray_moments(gray)\n").replace(after, "brightness")
    exec(compile(modified, "<original_quality_histogram_moments>", "exec"), namespace)
    implementation = namespace[original.__name__]
    trace = {"histogram_calls": 0, "native_calls": 0, "setup_seconds": time.perf_counter()-started}

    def quality(image):
        supported = (cv2.__version__ == "4.13.0" and isinstance(image, np.ndarray) and image.dtype == np.uint8
            and image.ndim == 3 and image.shape[2] == 3 and image.shape[0] > 0 and image.shape[1] > 0
            and image.shape[0]*image.shape[1] <= PIXEL_LIMIT)
        trace["histogram_calls" if supported else "native_calls"] += 1
        return implementation(image) if supported else original(image)

    return quality, trace


@contextmanager
def histogram_quality_execution(enabled=True):
    import quality_engine as native
    import run_initial_pipeline as entry
    original, old_entry = native.compact_quality, entry.compact_quality
    if original is not old_entry:
        raise ValueError("The original raw quality aliases have different ownership")
    quality, trace = make_quality(original)
    started = time.perf_counter()
    if enabled:
        native.compact_quality, entry.compact_quality = quality, quality
    trace["binding_setup_seconds"] = time.perf_counter()-started
    try:
        yield trace
    finally:
        started = time.perf_counter()
        native.compact_quality, entry.compact_quality = original, old_entry
        trace["restore_seconds"] = time.perf_counter()-started
        trace["aliases_restored"] = native.compact_quality is original and entry.compact_quality is old_entry
