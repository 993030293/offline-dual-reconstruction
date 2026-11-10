"""Compare complete array admission, native F results, and caller RNG state."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
import cv2
import numpy as np

from array_match_geometry import ArrayMatchGeometry, array_match_geometry
from check_parallel_neighbor_geometry import fixture, rng_probe
from integer_sift_matching import ExactMatcher
from observation_index import indexed_match
from runtime import atomic_json, data_identity

NATIVE_MATCHER = cv2.BFMatcher


def prepare(values):
    """Build the established exact adapter payload from native fixture matches."""
    bank, model, keypoints, descriptors, neighbors, index = values
    rows = []
    for image_id in neighbors:
        train = bank.original[image_id][2]
        matcher = NATIVE_MATCHER(cv2.NORM_L2)
        nearest = matcher.knnMatch(descriptors, train, k=2)
        backward = matcher.match(train, descriptors)
        forward = np.empty((len(descriptors), 4), np.int32)
        distances = np.empty((len(descriptors), 2), np.float32)
        reverse = np.empty(len(train), np.int32)
        if len(nearest) != len(descriptors):
            raise ValueError("A fixture must preserve the owned forward2 payload shape")
        for i, pair in enumerate(nearest):
            if len(pair) != 2:
                raise ValueError("A fixture requires two native train descriptors")
            forward[i] = pair[0].trainIdx, 0, pair[1].trainIdx, 0
            distances[i] = pair[0].distance, pair[1].distance
        for match in backward:
            reverse[match.queryIdx] = match.trainIdx
        rows.append((descriptors.copy(), train.copy(), (forward, distances, reverse)))
    return rows


@contextmanager
def prepared_factory(prepared):
    original, created = cv2.BFMatcher, []
    def factory(norm):
        if norm != cv2.NORM_L2:
            raise ValueError("The original norm changed")
        matcher = ExactMatcher(None, prepared)
        created.append(matcher)
        return matcher
    cv2.BFMatcher = factory
    try:
        yield created
    finally:
        cv2.BFMatcher = original


def invoke(function, values, geometric, prepared):
    cv2.setRNGSeed(7129)
    with prepared_factory(prepared) as created:
        try:
            result, error = function(*values, geometric=geometric), None
        except Exception as failure:
            result, error = None, (type(failure).__name__, str(failure))
        return result, error, rng_probe(), created


def equal(a, b):
    return a[:2] == b[:2] and np.array_equal(a[2], b[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if cv2.__version__ != "4.13.0":
        raise RuntimeError("The existing Windows geometry runtime is required")
    paths = [Path(__file__), Path(__file__).with_name("array_match_geometry.py"),
        Path(__file__).with_name("integer_sift_matching.py"),
        Path(__file__).with_name("check_parallel_neighbor_geometry.py"),
        ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/observation_index.py"]
    sources = [data_identity(p) for p in paths]
    rows = []
    for shape in ("curved", "thin", "irregular"):
        for scale, aspect in ((.01, .7), (1., 1.), (100., 1.9)):
            for count, neighbors in ((0, 3), (7, 3), (15, 3), (89, 1), (89, 4)):
                values = list(fixture(count, scale, aspect, shape, neighbors))
                for dtype in (np.float32, np.float64):
                    values[2] = values[2].astype(dtype)
                    payload = prepare(values)
                    for geometric in (False, True):
                        operator = ArrayMatchGeometry(indexed_match)
                        old = invoke(indexed_match, values, geometric, payload)
                        new = invoke(operator, values, geometric, payload)
                        rows.append({"shape": shape, "scale": scale, "aspect": aspect,
                            "count": count, "neighbors": neighbors, "dtype": str(dtype),
                            "geometric": geometric, "complete_outputs_exact": old[:2] == new[:2],
                            "caller_RNG_exact": np.array_equal(old[2], new[2]),
                            "actual_array_route": operator.trace["routes"] == ["owned_exact_array"],
                            "complete_payload_consumed": all(not m.prepared and m.pending is None for m in new[3])})
    checks = {"all180_complete_outputs_exact": len(rows) == 180 and all(r["complete_outputs_exact"] for r in rows),
        "all180_caller_RNG_exact": all(r["caller_RNG_exact"] for r in rows),
        "all180_actual_array_route_and_complete_consumption": all(r["actual_array_route"] and r["complete_payload_consumed"] for r in rows)}
    values = list(fixture(89, 1, 1, "irregular", 4))
    payload = prepare(values)
    for row in payload:
        row[2][1][:5] = np.asarray([[.7, .9], [.7, .875], [0., 0.], [.2, .25], [.20000002, .25]], np.float32)
    old = invoke(indexed_match, values, False, payload)
    new = invoke(ArrayMatchGeometry(indexed_match), values, False, payload)
    checks["strict_float32_distance_ratio_ties_and_duplicate_point_order_exact"] = equal(old, new)
    fit_original = cv2.findFundamentalMat
    try:
        for tag, reply in (("none_F", (None, None)), ("none_mask", (np.eye(3), None)),
            ("short_mask_original_zip", (np.eye(3), np.ones((16, 1), np.uint8)))):
            cv2.findFundamentalMat = lambda *args, answer=reply: answer
            payload = prepare(values)
            old = invoke(indexed_match, values, True, payload)
            new = invoke(ArrayMatchGeometry(indexed_match), values, True, payload)
            checks[tag+"_exact"] = equal(old, new)
        calls = []
        def fail(*args):
            calls.append(1)
            raise RuntimeError("owned_F_failure")
        cv2.findFundamentalMat = fail
        old = invoke(indexed_match, values, True, payload)
        new = invoke(ArrayMatchGeometry(indexed_match), values, True, payload)
        checks["F_exception_original_order_no_retry"] = equal(old, new) and old[1] == ("RuntimeError", "owned_F_failure") and len(calls) == 2
    finally:
        cv2.findFundamentalMat = fit_original
    for tag in ("pixel_mismatch", "observation_count", "prepared_order"):
        values = list(fixture(89, 1, 1, "curved", 3))
        payload = prepare(values)
        if tag == "pixel_mismatch":
            values[0].original[1][1][0, 0] += 2
        elif tag == "observation_count":
            name, kp, desc = values[0].original[1]
            values[0].original[1] = (name, kp[:-1], desc)
        else:
            payload[0], payload[1] = payload[1], payload[0]
            payload[0][1][0, 0] += np.float32(1/512)
        old = invoke(indexed_match, values, True, payload)
        new = invoke(ArrayMatchGeometry(indexed_match), values, True, payload)
        checks[tag+"_same_exception_and_RNG"] = equal(old, new) and old[1] is not None
    values = list(fixture(89, 1, 1, "thin", 4))
    payload = prepare(values)
    operator = ArrayMatchGeometry(indexed_match, enabled=False)
    old, new = invoke(indexed_match, values, True, payload), invoke(operator, values, True, payload)
    checks["disabled_original_before_consumption"] = equal(old, new) and operator.trace["routes"] == ["original_before_consumption"]
    values[2] = values[2].astype(np.float16)
    operator = ArrayMatchGeometry(indexed_match)
    old, new = invoke(indexed_match, values, False, payload), invoke(operator, values, False, payload)
    checks["unsupported_dtype_original_before_consumption"] = equal(old, new) and operator.trace["routes"] == ["original_before_consumption"]
    values = list(fixture(89, 1, 1, "curved", 3))
    cv2.setRNGSeed(7129)
    old, probe = indexed_match(*values, geometric=True), rng_probe()
    cv2.setRNGSeed(7129)
    operator = ArrayMatchGeometry(indexed_match)
    new, new_probe = operator(*values, geometric=True), rng_probe()
    checks["native_BF_unsupported_original_complete_outputs_and_RNG"] = old == new and np.array_equal(probe, new_probe) and operator.trace["routes"] == ["original_before_consumption"]
    import native_observation_index
    original = native_observation_index.indexed_match
    try:
        with array_match_geometry() as trace:
            raise RuntimeError("owned_context_failure")
    except RuntimeError as error:
        checks["context_failure_restores_hook"] = str(error) == "owned_context_failure" and native_observation_index.indexed_match is original and trace["binding_restored"]
    checks["all_sources_unchanged"] = sources == [data_identity(p) for p in paths]
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "rows": rows, "sources": sources, "whole_seconds": time.perf_counter()-STARTED,
        "fixture_payload_is_native_BF_not_live_GPU": True, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": checks, "whole_seconds": result["whole_seconds"]}, flush=True)
    if not all(checks.values()):
        raise AssertionError("A complete array assembly contract failed")


if __name__ == "__main__":
    main()
