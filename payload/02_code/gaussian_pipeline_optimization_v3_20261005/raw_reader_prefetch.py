"""Let one owned RAW generator run ahead of the unchanged bounded scoring pool."""

from contextlib import contextmanager
import queue
import threading
import time


class OwnedRawPrefetch:
    def __init__(self, source, capacity=2):
        if type(capacity) is not int or capacity != 2:
            raise ValueError("The private RAW reader requires its fixed two-result bound")
        self.source = source
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.thread = None
        self.closed = False
        self.used = False
        self.trace = {"capacity": capacity, "source_next_calls": 0, "produced": 0, "delivered": 0,
            "peak_queue_results": 0, "reader_thread": None, "source_closed_on_reader": False,
            "source_error": None, "closed": False, "reader_alive_after_join": None,
            "producer_read_seconds": 0., "consumer_wait_seconds": 0., "shutdown_seconds": 0.}

    def put(self, tag, value):
        while not self.stop.is_set():
            try:
                self.queue.put((tag, value), timeout=.01)
                self.trace["peak_queue_results"] = max(self.trace["peak_queue_results"], self.queue.qsize())
                return True
            except queue.Full:
                continue
        return False

    def produce(self):
        self.trace["reader_thread"] = threading.get_ident()
        iterator, failure = None, None
        try:
            iterator = iter(self.source)
            while not self.stop.is_set():
                began = time.perf_counter()
                try:
                    self.trace["source_next_calls"] += 1
                    item = next(iterator)
                except StopIteration:
                    break
                finally:
                    self.trace["producer_read_seconds"] += time.perf_counter() - began
                self.trace["produced"] += 1
                if not self.put("value", item):
                    break
        except BaseException as error:
            failure = error
            self.trace["source_error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            try:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
                self.trace["source_closed_on_reader"] = True
            except BaseException as error:
                self.trace["source_close_error"] = {"type": type(error).__name__, "message": str(error)}
                if failure is None:
                    failure = error
            self.put("error" if failure is not None else "end", failure)

    def __iter__(self):
        if self.closed or self.used:
            raise RuntimeError("An owned RAW prefetch stream cannot be reused")
        self.used = True
        self.thread = threading.Thread(target=self.produce, name="owned-RAW-reader", daemon=False)
        self.thread.start()
        try:
            while True:
                began = time.perf_counter()
                tag, value = self.queue.get()
                self.trace["consumer_wait_seconds"] += time.perf_counter() - began
                if tag == "error":
                    raise value
                if tag == "end":
                    return
                self.trace["delivered"] += 1
                yield value
        finally:
            self.close()

    def close(self):
        if not self.closed:
            began = time.perf_counter()
            self.stop.set()
            if self.thread is not None:
                self.thread.join()
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            self.source = None
            self.closed = True
            self.trace.update(closed=True, reader_alive_after_join=bool(self.thread and self.thread.is_alive()),
                queued_references_after_close=self.queue.qsize(), shutdown_seconds=time.perf_counter() - began)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()


@contextmanager
def raw_reader_prefetch_execution(enabled=True):
    import quality_engine as native
    import run_initial_pipeline as entry
    original, original_entry = native.score_messages, entry.score_messages
    if original is not original_entry:
        raise ValueError("The original RAW scoring aliases have different owners")
    trace = {"enabled": enabled, "calls": [], "setup_seconds": 0., "restore_seconds": 0., "binding_restored": False}

    def score(messages, *values, **options):
        with OwnedRawPrefetch(messages) as producer:
            trace["calls"].append(producer.trace)
            return original(iter(producer), *values, **options)

    began = time.perf_counter()
    if enabled:
        native.score_messages = entry.score_messages = score
    trace["setup_seconds"] = time.perf_counter() - began
    try:
        yield trace
    finally:
        began = time.perf_counter()
        if enabled and (native.score_messages is not score or entry.score_messages is not score):
            raise RuntimeError("Another writer changed the private RAW scoring binding")
        native.score_messages, entry.score_messages = original, original_entry
        trace["binding_restored"] = native.score_messages is original and entry.score_messages is original_entry
        trace["restore_seconds"] = time.perf_counter() - began
