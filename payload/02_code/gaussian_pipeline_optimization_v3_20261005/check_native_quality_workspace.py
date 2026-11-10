"""Check complete native quality, private planes, errors, and thread ownership."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import sys
import threading

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from quality_engine import compact_quality
import quality_histogram as histogram
from native_quality_workspace import native_quality_workspace_execution


def outcome(function, image):
    try:
        return "value", function(image)
    except cv2.error as error:
        return "cv2_error", (error.code, error.func, error.err)
    except (TypeError, ValueError) as error:
        return type(error).__name__, str(error)


def owner(function):
    implementation = inspect.getclosurevars(function).nonlocals["implementation"]
    return implementation.__globals__["_native_quality_workspace"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("native_quality_workspace.py"),
             Path(histogram.__file__), V1 / "quality_engine.py"]
    sources = [data_identity(path) for path in paths]
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(paths):
        (snapshot / (str(index) + "_" + path.name)).write_bytes(path.read_bytes())
    atomic_json(args.output / "CONTRACT.json", {"sources": sources, "no_quality_formula_or_native_parameter_change": True,
        "thread_count": 4, "unsupported_routes_original_before_workspace": True, "counted": False})
    checks, rows, error_traces = {}, [], []
    reference, _ = histogram.make_quality(compact_quality)
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261006)
    original_builder = histogram.make_quality
    try:
        with native_quality_workspace_execution(True) as single:
            candidate, candidate_trace = histogram.make_quality(compact_quality)
            workspace = owner(candidate)
            images = [np.full((32, 48, 3), value, np.uint8) for value in (0, 1, 5, 6, 127, 249, 250, 254, 255)]
            for shape in ((1, 1, 3), (1, 7, 3), (7, 1, 3), (2, 2, 3), (31, 43, 3), (80, 64, 3), (577, 1009, 3)):
                image = rng.integers(0, 256, shape, np.uint8)
                images.extend((image, np.asfortranarray(image), image[:, ::-1], image.copy()))
            for index, image in enumerate(images):
                a, b = outcome(reference, image), outcome(candidate, image)
                exact_planes = True
                if a[0] == "value":
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    derivative = cv2.Laplacian(gray, cv2.CV_16S)
                    reduced = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
                    state = workspace.local.state
                    exact_planes = all(np.array_equal(state[name], value) for name, value in
                        (("gray", gray), ("derivative", derivative), ("reduced", reduced)))
                rows.append({"index": index, "shape": list(image.shape), "full_native_quality_and_error_exact": a == b,
                             "all_three_native_planes_exact": exact_planes})
            unsupported = []
            for image in (np.zeros((0, 3, 3), np.uint8), np.zeros((8, 8, 3), np.uint16),
                          np.zeros((8, 8, 3), np.float32), np.zeros((8, 8), np.uint8),
                          np.zeros((8, 8, 4), np.uint8), None):
                before = candidate_trace["native_calls"]
                unsupported.append(outcome(reference, image) == outcome(candidate, image)
                                   and candidate_trace["native_calls"] == before + 1)
        checks["all37_quality_size_layout_change_and_planes_exact"] = len(rows) == 37 and all(
            row["full_native_quality_and_error_exact"] and row["all_three_native_planes_exact"] for row in rows)
        checks["all6_unsupported_native_values_and_errors_preserved"] = len(unsupported) == 6 and all(unsupported)
        checks["single_thread_planes_reused_and_all_references_cleared"] = single["workspace"]["reused"] > 0 and single[
            "workspace"]["closed"] and single["workspace"]["owned_plane_bytes_after_close"] == 0 and single["original_builder_restored"]
        with native_quality_workspace_execution(True) as threaded:
            candidate, _ = histogram.make_quality(compact_quality)
            workspace = owner(candidate)
            barrier = threading.Barrier(4)

            def thread(index):
                barrier.wait(timeout=10)
                equal = True
                for repeat in range(6):
                    image = rng.integers(0, 256, (61 + index, 83 + index, 3), np.uint8)
                    equal &= outcome(reference, image) == outcome(candidate, image)
                state = workspace.local.state
                return {"owner_thread": threading.get_ident(), "equal": bool(equal),
                        "pointers": [int(state[name].ctypes.data) for name in ("gray", "derivative", "reduced")]}

            with ThreadPoolExecutor(max_workers=4) as pool:
                thread_rows = list(pool.map(thread, range(4)))
        pointers = [pointer for row in thread_rows for pointer in row["pointers"]]
        checks["four_actual_thread_owners_24_scores_and12_planes_disjoint"] = len({row["owner_thread"] for row in thread_rows}) == 4 and all(
            row["equal"] for row in thread_rows) and len(set(pointers)) == 12 and threaded["workspace"]["thread_count"] == 4
        checks["all_thread_planes_cleared_after_native_worker_drain"] = threaded["original_builder_restored"] and threaded[
            "workspace"]["closed"] and threaded["workspace"]["owned_plane_bytes_after_close"] == 0
        for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
            try:
                with native_quality_workspace_execution(True) as trace:
                    with histogram.histogram_quality_execution(True):
                        import quality_engine
                        quality_engine.compact_quality(np.full((64, 80, 3), 127, np.uint8))
                        raise error_type("injected_private_quality_consumer_failure")
            except error_type as error:
                if str(error) != "injected_private_quality_consumer_failure":
                    raise
                error_traces.append(trace)
        checks["RuntimeError_KeyboardInterrupt_SystemExit_original_builder_and_planes_restored"] = len(error_traces) == 3 and all(
            row["original_builder_restored"] and row["workspace"]["closed"] and row["workspace"]["owned_plane_bytes_after_close"] == 0 for row in error_traces)

        def unbound_builder(original):
            raise RuntimeError("A changed source must never execute")

        histogram.make_quality = unbound_builder
        refused = False
        try:
            with native_quality_workspace_execution(True):
                pass
        except ValueError as error:
            refused = "anchors changed" in str(error)
        finally:
            histogram.make_quality = original_builder
        checks["source_anchor_change_refused_before_native_execution"] = refused
    finally:
        histogram.make_quality = original_builder
        cv2.setNumThreads(old_threads)
    checks["all_original_sources_builder_and_OpenCV_threads_restored"] = sources == [data_identity(path) for path in paths] and histogram.make_quality is original_builder and cv2.getNumThreads() == old_threads
    result = {"status": "generic_passed_pilot_and_complete_stage_pending" if all(checks.values()) else "failed_verification",
              "checks": checks, "sources": sources, "rows": rows, "thread_rows": thread_rows,
              "single_trace": single, "threaded_trace": threaded, "error_traces": error_traces,
              "counted": False, "accepted_count_contribution": 0}
    atomic_json(args.output / "RESULT.json", result)
    print({"status": result["status"], "checks": checks, "accepted_count_contribution": 0}, flush=True)
    if not all(checks.values()):
        raise ValueError("A complete original native quality workspace contract failed")


if __name__ == "__main__":
    main()
