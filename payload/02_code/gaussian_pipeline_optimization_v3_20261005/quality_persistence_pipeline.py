"""Drain original durable score writes before returning the complete scoring call."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import sys
import threading
import time


class QualityWriter:
    def __init__(self, capacity=64):
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("The owned quality writer queue must be bounded")
        self.capacity = capacity
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owned-quality-write")
        self.slots = threading.BoundedSemaphore(capacity)
        self.lock = threading.RLock()
        self.error = None
        self.closed = False
        self.trace = {"capacity": capacity, "workers": 1, "submitted": 0, "finished": 0,
            "committed": 0, "maximum_pending": 0, "write_seconds_sum": 0., "close_seconds": 0., "closed": False}

    def persist(self, function, args):
        with self.lock:
            if self.error is not None:
                raise self.error
        started = time.perf_counter()
        try:
            result = function(*args)
            with self.lock:
                self.trace["committed"] += 1
            return result
        finally:
            with self.lock:
                self.trace["write_seconds_sum"] += time.perf_counter()-started

    def completed(self, future):
        with self.lock:
            error = future.exception()
            if error is not None and self.error is None:
                self.error = error
            self.trace["finished"] += 1
        self.slots.release()

    def submit(self, function, cache, key, scores):
        with self.lock:
            if self.closed:
                raise RuntimeError("An owned quality writer cannot accept after close")
            if self.error is not None:
                raise self.error
        snapshot = dict(scores)
        self.slots.acquire()
        with self.lock:
            if self.closed or self.error is not None:
                self.slots.release()
                if self.error is not None:
                    raise self.error
                raise RuntimeError("An owned quality writer closed during submission")
            try:
                future = self.pool.submit(self.persist, function, (cache, key, snapshot))
            except BaseException:
                self.slots.release()
                raise
            self.trace["submitted"] += 1
            self.trace["maximum_pending"] = max(self.trace["maximum_pending"], self.trace["submitted"]-self.trace["finished"])
            future.add_done_callback(self.completed)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        started = time.perf_counter()
        try:
            self.pool.shutdown(wait=True, cancel_futures=False)
            with self.lock:
                if self.trace["submitted"] != self.trace["finished"]:
                    raise ValueError("The original durable cache writes were not all drained")
                if self.error is not None:
                    raise self.error
                if self.trace["committed"] != self.trace["submitted"]:
                    raise ValueError("A complete original cache write was not committed")
        finally:
            self.trace.update(closed=True, close_seconds=time.perf_counter()-started)


@contextmanager
def quality_persistence_execution(enabled, owned_cache_root):
    started = time.perf_counter()
    import quality_engine as native
    import run_initial_pipeline as entry
    old_score, old_entry = native.score_messages, entry.score_messages
    original_write = native.ScoreCache.write
    if old_score is not old_entry:
        raise ValueError("The original scoring aliases have different ownership")
    signature = inspect.signature(old_score)
    trace = {"enabled": enabled, "calls": [], "setup_seconds": None, "restore_seconds": None}
    active = False

    def scoring(*args, **kwargs):
        nonlocal active
        bound = signature.bind(*args, **kwargs)
        cache = bound.arguments.get("cache")
        if cache is None:
            return old_score(*args, **kwargs)
        if type(cache) is not native.ScoreCache or cache.directory.resolve() != owned_cache_root.resolve() or active:
            raise ValueError("An async scoring call must own its private original cache")
        active = True
        before = time.perf_counter()
        writer = QualityWriter()
        writer.trace["initial_setup_seconds"] = time.perf_counter()-before
        trace["calls"].append(writer.trace)
        native.ScoreCache.write = lambda store, key, scores: writer.submit(original_write, store, key, scores)
        try:
            return old_score(*args, **kwargs)
        finally:
            main_error = sys.exception()
            try:
                writer.close()
            except BaseException as cleanup_error:
                writer.trace["error"] = repr(cleanup_error)
                if main_error is None:
                    raise
                main_error.add_note("Owned quality persistence also failed: "+repr(cleanup_error))
            finally:
                native.ScoreCache.write = original_write
                active = False

    if enabled:
        native.score_messages, entry.score_messages = scoring, scoring
    trace["setup_seconds"] = time.perf_counter()-started
    try:
        yield trace
    finally:
        started = time.perf_counter()
        native.score_messages, entry.score_messages = old_score, old_entry
        native.ScoreCache.write = original_write
        trace["restore_seconds"] = time.perf_counter()-started
        trace["original_functions_restored"] = native.score_messages is old_score and entry.score_messages is old_entry and native.ScoreCache.write is original_write
