"""Check unchanged quality moments on every image in the external scene workspace."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from quality_engine import compact_quality
from quality_histogram import make_quality


def score_all(paths, identities, function):
    def process(path):
        if data_identity(path) != identities[str(path)]:
            raise ValueError("An external complete image changed before scoring")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or data_identity(path) != identities[str(path)]:
            raise ValueError("An external complete image was unreadable or changed during scoring")
        return {"name": path.name, "shape": list(image.shape), "quality": function(image)}
    iterator, pending, rows = iter(paths), deque(), []
    exhausted = False
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="external-quality") as pool:
        try:
            while not exhausted or pending:
                while not exhausted and len(pending) < 16:
                    try:
                        path = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(pool.submit(process, path))
                if pending:
                    rows.append(pending.popleft().result())
        finally:
            for future in pending:
                future.cancel()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.image_root.glob("*.png"))
    if len(paths) != args.expected_images or len(paths) == 0:
        raise ValueError("The complete external image denominator changed")
    sources = [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("quality_histogram.py"), V1/"quality_engine.py")]
    identities = {str(path): data_identity(path) for path in paths}
    state = {"status": "running", "sources": sources, "input_images": list(identities.values()),
        "expected_complete_image_count": args.expected_images, "three_pairs": [], "new_optimization_count": 0,
        "component_reads_decodes_both_full_image_identities_all_quality_fields_first_setup_pool_and_close": True,
        "workers": 4, "queue_capacity": 16, "minimum_component_reduction": .1,
        "old_quality_formula_pixels_and_all_image_denominators_unchanged": True,
        "no_GT_TLS_map_pose_ROI_frame_selection_or_object_label_input": True,
        "external_component_is_not_raw_full_sequence_selection_or_geometry_quality": True,
        "Oxford_already_development_no_strict_blind_evaluation": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    old = cv2.getNumThreads()
    cv2.setNumThreads(1)
    started = time.perf_counter()
    try:
        for trial in range(3):
            order = ("native", "histogram") if trial % 2 == 0 else ("histogram", "native")
            durations, outputs, traces = {}, {}, {}
            for mode in order:
                before = time.perf_counter()
                function, trace = make_quality(compact_quality) if mode == "histogram" else (compact_quality, None)
                outputs[mode] = score_all(paths, identities, function)
                durations[mode] = time.perf_counter()-before
                traces[mode] = trace
            checks = {"all_original_complete360_image_pixels_identities_shapes_and_quality_fields":
                len(outputs["native"]) == len(outputs["histogram"]) == args.expected_images
                and outputs["native"] == outputs["histogram"],
                "all_actual_images_use_the_admitted_quality_format": traces["histogram"]["histogram_calls"] == args.expected_images
                    and traces["histogram"]["native_calls"] == 0,
                "complete_component_pair_cannot_regress_more10percent": durations["histogram"] <= 1.1*durations["native"]}
            row = {"trial": trial, "order": order, "complete_component_seconds": durations, "checks": checks, "trace": traces["histogram"]}
            state["three_pairs"].append(row)
            atomic_json(args.output/"STATE.json", state)
            atomic_json(args.output/f"trial{trial}_NATIVE_RECORDS.json", outputs["native"])
            atomic_json(args.output/f"trial{trial}_HISTOGRAM_RECORDS.json", outputs["histogram"])
            print(row, flush=True)
            if not all(checks.values()):
                raise ValueError("An external complete quality field or component cost guard failed")
        median = {mode: statistics.median(row["complete_component_seconds"][mode] for row in state["three_pairs"])
            for mode in ("native", "histogram")}
        reduction = 1-median["histogram"]/median["native"]
        checks = {"all_three_complete_external_image_components_protected": len(state["three_pairs"]) == 3
                and all(all(row["checks"].values()) for row in state["three_pairs"]),
            "complete_component_reduces10percent": reduction >= .1,
            "source_unchanged": sources == [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("quality_histogram.py"), V1/"quality_engine.py")]}
        state.update(status="external_component_passed" if all(checks.values()) else "failed_verification", checks=checks,
            median=median, reduction=reduction, elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        cv2.setNumThreads(old)


if __name__ == "__main__":
    main()
