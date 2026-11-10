"""Prefetch one query's reference preparation while retaining serial pose estimation."""

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import threading
import time


class OrderedQueryPreparation:
    def __init__(self, parallel=True, capacity=2):
        if type(parallel) is not bool or type(capacity) is not int or not 1 <= capacity <= 2:
            raise ValueError("Query reference preparation requires one bounded producer")
        self.parallel, self.capacity = parallel, capacity
        self.failed = threading.Event()
        self.pool = None
        self.pending = deque()
        self.closed = False
        self.trace = {"parallel": parallel, "capacity": capacity, "workers": 1,
            "submitted": 0, "completed": 0, "delivered": 0, "peak_pending": 0,
            "preparations": [], "closed": False, "shutdown_seconds": 0.}

    def produce(self, ordinal, prepare, source):
        if self.failed.is_set():
            raise CancelledError("An earlier owned query preparation already failed")
        started = time.perf_counter()
        try:
            value = prepare(source)
            self.trace["completed"] += 1
            return value
        except BaseException:
            self.failed.set()
            raise
        finally:
            self.trace["preparations"].append({"ordinal": ordinal, "start": started,
                "end": time.perf_counter(), "thread": threading.get_ident()})

    def iterate(self, sources, prepare):
        if self.closed or self.trace["submitted"]:
            raise RuntimeError("An owned query producer cannot be reused")
        iterator = iter(enumerate(sources))
        try:
            if not self.parallel:
                for ordinal, source in iterator:
                    self.trace["submitted"] += 1
                    value = self.produce(ordinal, prepare, source)
                    self.trace["delivered"] += 1
                    yield value
                return
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owned-query-reference")
            exhausted = False
            while not exhausted or self.pending:
                while not exhausted and len(self.pending) < self.capacity:
                    try:
                        ordinal, source = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    self.pending.append(self.pool.submit(self.produce, ordinal, prepare, source))
                    self.trace["submitted"] += 1
                    self.trace["peak_pending"] = max(self.trace["peak_pending"], len(self.pending))
                if self.pending:
                    value = self.pending.popleft().result()
                    self.trace["delivered"] += 1
                    yield value
        finally:
            self.close()

    def close(self):
        if not self.closed:
            started = time.perf_counter()
            for future in self.pending:
                future.cancel()
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()
            self.closed = True
            self.trace.update(closed=True, shutdown_seconds=time.perf_counter()-started)


@contextmanager
def query_preparation_pipeline(mode):
    started = time.perf_counter()
    import localize_new_map_evaluation as native
    if mode not in ("incumbent", "serial", "parallel"):
        raise ValueError("An unsupported owned query preparation route was selected")
    original = native.main
    source = inspect.getsource(original)
    first = "        for source in evaluation_rows:\n"
    after = "        if not np.array_equal(original_parameters, camera.params) or identities != [data_identity(path) for path in source_paths]:\n"
    branch = '            if not rough["coarse_pose_valid"]:\n'
    if source.count(first) != 1 or source.count(after) != 1 or source.count(branch) != 1:
        raise ValueError("The pinned original query-loop boundaries changed")
    prefix, rest = source.split(first)
    body, suffix = rest.split(after)
    prep, calculation = body.split(branch)
    feature = "                keypoints, descriptors = bank.extra(image_path)\n"
    if calculation.count(feature) != 1 or calculation.count("            results.append(record)\n") != 1:
        raise ValueError("The original query feature and commit ownership changed")
    producer = None if mode == "incumbent" else OrderedQueryPreparation(mode == "parallel")
    trace = {"mode": mode, "consumer_intervals": [], "query_loop_seconds": None,
        "composition_seconds": None, "producer": producer.trace if producer else None,
        "only_reference_preparation_moved_SIFT_map_matches_PnP_and_commits_stay_on_caller": True}
    calculation = calculation.replace("            results.append(record)\n",
        '            _query_trace["consumer_intervals"].append({"ordinal": len(_query_trace["consumer_intervals"]), '
        '"start": _consumer_started, "end": time.perf_counter(), "thread": _caller_thread.get_ident()})\n'
        "            results.append(record)\n")
    consumer = "            _consumer_started = time.perf_counter()\n"+branch+calculation
    if producer is None:
        loop = first+prep+consumer
    else:
        function = "        def _prepare_reference(source):\n"+prep
        function += "            return source, name, rough, image_path, base, query_started\n"
        loop = function+'        _prepared_queries = _query_producer.iterate(evaluation_rows, _prepare_reference)\n'
        loop += "        for source, name, rough, image_path, base, query_started in _prepared_queries:\n"+consumer
    lines = loop.splitlines(keepends=True)
    wrapped = '        _query_loop_started = time.perf_counter()\n        try:\n'+"".join("    "+line for line in lines)
    wrapped += "        finally:\n"
    if producer is not None:
        wrapped += "            _query_producer.close()\n"
    wrapped += '            _query_trace["query_loop_seconds"] = time.perf_counter()-_query_loop_started\n'
    namespace = dict(native.__dict__)
    namespace.update(_query_trace=trace, _query_producer=producer, _caller_thread=threading)
    # Resolve the existing immutable identity adapter after its outer context binds it.
    namespace["data_identity"] = lambda path: native.data_identity(path)
    exec(compile(prefix+wrapped+after+suffix, "<original_query_reference_preparation_pipeline>", "exec"), namespace)
    native.main = namespace["main"]
    trace["composition_seconds"] = time.perf_counter()-started
    try:
        yield trace
    finally:
        native.main = original
        if producer is not None:
            producer.close()
