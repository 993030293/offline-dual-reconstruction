"""Overlap one current-frame descriptor task with unchanged CPU geometry checks."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import sys
import time

import cv2
import numpy as np

from generic_sift_matching import GenericWorker, admission
from integer_sift_matching import DistanceWorker, ExactMatcher, integer_sift_matching


@contextmanager
def paired_match_pipeline(log, mode="parallel", worker_kind="generic", enabled=True):
    if mode not in ("serial", "parallel") or worker_kind not in ("legacy", "generic"):
        raise ValueError("The explicit distance pipeline configuration is unsupported")
    import feature_localization
    import localize_extra_frames
    original_match, original_split, original_factory = feature_localization.FeatureBank.match, localize_extra_frames.split_correspondences, cv2.BFMatcher
    pending, active, worker = [], [], None
    pool = ThreadPoolExecutor(max_workers=1) if mode == "parallel" and enabled else None
    trace = {"mode": mode, "worker_kind": worker_kind, "splits": [], "match_calls": 0,
        "maximum_pending_distance_tasks": 1 if pool else 0, "only_parent_reads_SQLite": True,
        "new_optimization_count": 0}
    began = time.perf_counter()

    def batch(query, trains, record):
        record["start"] = time.perf_counter()-began
        try:
            return worker.batch(query, trains)
        finally:
            record["end"] = time.perf_counter()-began

    def factory(norm):
        if norm != cv2.NORM_L2:
            raise ValueError("The unchanged matcher requires its original native L2 norm")
        return ExactMatcher(worker, active[0]) if active else original_factory(norm)

    def match(bank, model, keypoints, descriptors, neighbors, geometric=True):
        if not pending:
            return original_match(bank, model, keypoints, descriptors, neighbors, geometric)
        item = pending.pop(0)
        if item["neighbors"] != neighbors or not np.array_equal(descriptors, item["query"]):
            raise ValueError("A paired distance task changed query values or neighbor order")
        values = item["result"]
        if values is None:
            values = item["future"].result() if item["future"] is not None else batch(item["query"], item["trains"], item["distance_task"])
        active.append([(descriptors, t, r) for t, r in zip(item["trains"], values)])
        item["geometry_task"]["start"] = time.perf_counter()-began
        try:
            trace["match_calls"] += 1
            return original_match(bank, model, keypoints, descriptors, neighbors, geometric)
        finally:
            item["geometry_task"]["end"] = time.perf_counter()-began
            active.pop()

    def split(bank, model, keypoints, descriptors, neighbors):
        nonlocal worker
        if pending or active:
            raise ValueError("An owned pair cannot overlap another frame or share mutable state")
        queries = [descriptors[np.arange(parity, len(keypoints), 2)] for parity in (0, 1)]
        trains = []
        for image_id in neighbors:
            value = bank.original[image_id][2]
            trains.append(value.astype(np.float32)/512 if value.dtype == np.uint8 else value)
        routes = [admission(q, trains)[0] for q in queries]
        compatible = all(route == "integer" for route in routes)
        if worker_kind == "legacy" and (any(len(q) > 12000 for q in queries) or any(len(t) > 12000 for t in trains)):
            compatible = False
        if not compatible:
            trace["splits"].append({"route": "native_before_execution", "all_features_preserved": True})
            return original_split(bank, model, keypoints, descriptors, neighbors)
        if worker is None:
            worker = (DistanceWorker if worker_kind == "legacy" else GenericWorker)(log)
        tasks = [{"parity": parity, "distance_task": {}, "geometry_task": {}} for parity in (0, 1)]
        even_result = batch(queries[0], trains, tasks[0]["distance_task"])
        future = pool.submit(batch, queries[1], trains, tasks[1]["distance_task"]) if pool else None
        pending.extend([{"query": queries[0], "trains": trains, "neighbors": list(neighbors), "result": even_result,
            "future": None, **tasks[0]}, {"query": queries[1], "trains": trains, "neighbors": list(neighbors),
            "result": None, "future": future, **tasks[1]}])
        try:
            result = original_split(bank, model, keypoints, descriptors, neighbors)
            if pending:
                raise ValueError("The original complete pair did not consume both descriptor tasks")
            trace["splits"].append({"route": "integer", "tasks": tasks, "all_features_preserved": True,
                "overlap_seconds": min(tasks[0]["geometry_task"]["end"], tasks[1]["distance_task"]["end"])
                    -max(tasks[0]["geometry_task"]["start"], tasks[1]["distance_task"]["start"])})
            return result
        finally:
            error = sys.exception()
            try:
                if future is not None:
                    future.result()
            except BaseException as distance_error:
                if error is None:
                    raise
                error.add_note("The owned paired distance task also failed: "+repr(distance_error))
            finally:
                pending.clear()

    try:
        if enabled:
            feature_localization.FeatureBank.match = match
            localize_extra_frames.split_correspondences = split
            cv2.BFMatcher = factory
        yield trace
    finally:
        active_error = sys.exception()
        feature_localization.FeatureBank.match, localize_extra_frames.split_correspondences, cv2.BFMatcher = original_match, original_split, original_factory
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if worker is not None:
            try:
                worker.close()
            except BaseException as cleanup_error:
                trace["cleanup_error"] = repr(cleanup_error)
                if active_error is None:
                    raise
                active_error.add_note("Owned pair worker cleanup also failed: "+repr(cleanup_error))
            finally:
                trace["worker"] = worker.receipt
        trace["all_owned_tasks_complete"] = not pending and not active and (worker is None or worker.receipt["returncode"] == 0)


@contextmanager
def pipeline_context(log, mode, worker_kind, enabled=True):
    import localize_extra_frames
    context = integer_sift_matching(log, enabled) if mode == "incumbent" else paired_match_pipeline(log, mode, worker_kind, enabled)
    durations = []
    before = time.perf_counter()
    with context as trace:
        initialization = time.perf_counter()-before
        original = localize_extra_frames.split_correspondences

        def timed(*args, **kwargs):
            begin = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                durations.append(time.perf_counter()-begin)

        localize_extra_frames.split_correspondences = timed
        try:
            yield trace
        finally:
            localize_extra_frames.split_correspondences = original
            shutdown_started = time.perf_counter()
    shutdown = time.perf_counter()-shutdown_started
    trace["complete_current_frame_split_seconds"] = durations
    trace["initialization_and_shutdown_seconds"] = initialization+shutdown
    trace["complete_match_phase_seconds"] = sum(durations)+initialization+shutdown
