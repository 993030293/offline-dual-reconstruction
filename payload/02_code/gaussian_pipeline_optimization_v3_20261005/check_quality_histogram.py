"""Check generic quality contracts and a prespecified complete-scorer raw pilot."""

import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT/"02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
from quality_engine import compact_quality
from quality_histogram import make_quality, histogram_quality_execution


def outcome(function, image):
    try:
        return "value", function(image)
    except cv2.error as error:
        return "cv2_error", (error.code, error.func, error.err)
    except (ValueError, TypeError) as error:
        return type(error).__name__, str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    native = importlib.import_module("initial_selection")
    sys.path.insert(0, str(native.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    paths = [Path(__file__), Path(__file__).with_name("quality_histogram.py"), V1/"quality_engine.py",
        Path(native.__file__), native.TOOLS/"mcap_audit.py", native.TOOLS/"camera_motion.py", native.TOOLS/"color_contract.py"]
    sources = [data_identity(path) for path in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": sources, "pilot_frames": 128, "repetitions": 3,
        "minimum_complete_scorer_reduction": .1, "selection": "uniform_original_raw_order_excluded_IDs_removed_no_quality_pose_SfM_GT_ranking",
        "pixel_and_full_quality_fields_must_be_exact": True, "cached_RGB_read_cost_outside_both_pilot_timings": True,
        "cold_full7338_actual_entry_pending_and_pilot_cannot_count": True})
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    quality, trace = make_quality(compact_quality)
    rng = np.random.default_rng(20261005)
    rows = []
    try:
        for value in range(256):
            image = np.full((8, 12, 3), value, np.uint8)
            rows.append({"kind": "uniform", "value": value, "exact": outcome(compact_quality, image) == outcome(quality, image)})
        for shape in ((1, 1, 3), (1, 7, 3), (7, 1, 3), (2, 2, 3), (31, 43, 3), (80, 64, 3), (577, 1009, 3)):
            image = rng.integers(0, 256, shape, np.uint8)
            for layout, item in (("native", image), ("fortran", np.asfortranarray(image)), ("stride", image[:, ::-1])):
                rows.append({"kind": "layout", "shape": shape, "layout": layout,
                    "exact": outcome(compact_quality, item) == outcome(quality, item)})
        unsupported = []
        for item in (np.zeros((0, 3, 3), np.uint8), np.zeros((8, 8, 3), np.uint16),
                np.zeros((8, 8, 3), np.float32), np.zeros((8, 8), np.uint8), np.zeros((8, 8, 4), np.uint8)):
            before = trace["native_calls"]
            unsupported.append(outcome(compact_quality, item) == outcome(quality, item) and trace["native_calls"] == before+1)
        with histogram_quality_execution(True) as restored:
            try:
                raise RuntimeError("injected_quality_consumer_failure")
            finally:
                pass
    except RuntimeError as error:
        if str(error) != "injected_quality_consumer_failure":
            raise
        restore = restored["aliases_restored"]
    finally:
        cv2.setNumThreads(old_threads)
    generic = {"all256_gray_levels_and21_size_layout_oracles_exact": len(rows) == 277 and all(row["exact"] for row in rows),
        "all5_unsupported_types_and_errors_routed_before_execution": len(unsupported) == 5 and all(unsupported),
        "original_aliases_restored_after_failure": restore}
    atomic_json(args.output/"GENERIC.json", {"checks": generic, "rows": rows, "unsupported": unsupported,
        "sources": sources, "new_object_pipeline_quality_claim": False})
    if not all(generic.values()):
        raise ValueError("A generic complete original quality contract changed")
    exclusion = ROOT/"04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/EXCLUDED_EVALUATION_IDS.json"
    excluded = {int(name) for name in json.loads(exclusion.read_text(encoding="utf-8"))["frames"]}
    index = McapIndex(native.RAW)
    allowed = [frame for frame in range(index.message_count(native.CAMERA_TOPIC)) if frame not in excluded]
    selected = [allowed[i] for i in np.linspace(0, len(allowed)-1, 128, dtype=int)]
    images, identities = [], []
    for frame, message in index.iter_selected_messages(native.CAMERA_TOPIC, selected):
        stamp, optical, raw = decode_bayer(message["payload"])
        images.append(demosaic_rggb8_to_bgr(raw))
        identities.append({"frame": frame, "header_ns": stamp, "optical": optical})
    atomic_json(args.output/"PILOT_INPUT.json", {"selected_original_ids": selected, "actual_raw_metadata": identities,
        "exclusion_ID_source": data_identity(exclusion), "no_SfM_pose_or_GT_inputs": True})
    if [row["frame"] for row in identities] != selected or len(images) != 128:
        raise ValueError("The complete raw pilot denominator changed")
    pairs = []
    cv2.setNumThreads(1)
    try:
        for trial in range(3):
            order = ("native", "histogram") if trial % 2 == 0 else ("histogram", "native")
            outputs, durations = {}, {}
            for mode in order:
                started = time.perf_counter()
                function, setup = make_quality(compact_quality) if mode == "histogram" else (compact_quality, None)
                outputs[mode] = [function(image) for image in images]
                durations[mode] = time.perf_counter()-started
            row = {"trial": trial, "order": order, "complete_scorer_seconds_with_first_setup": durations,
                "all128_full_quality_records_exact": outputs["native"] == outputs["histogram"]}
            pairs.append(row)
            atomic_json(args.output/"STATE.json", {"status": "running", "pairs": pairs})
            if not row["all128_full_quality_records_exact"]:
                raise ValueError("A complete original raw quality score changed")
    finally:
        cv2.setNumThreads(old_threads)
    median = {mode: statistics.median(row["complete_scorer_seconds_with_first_setup"][mode] for row in pairs) for mode in ("native", "histogram")}
    reduction = 1-median["histogram"]/median["native"]
    checks = {**generic, "all_original128_raw_pilot_fields_exact": len(pairs) == 3 and all(row["all128_full_quality_records_exact"] for row in pairs),
        "complete_scorer_diagnostic_reduces10percent": reduction >= .1,
        "sources_unchanged": sources == [data_identity(path) for path in paths]}
    result = {"status": "pilot_passed_full_entry_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": sources, "pairs": pairs, "median": median, "reduction": reduction,
        "generic_trace": trace, "counted": False, "actual_full7338_cold_selection_not_run": True,
        "cached_RGB_read_excluded_only_from_this_diagnostic_no_whole_entry_claim": True}
    atomic_json(args.output/"RESULT.json", result)
    print({key: result[key] for key in ("status", "checks", "median", "reduction")}, flush=True)


if __name__ == "__main__":
    main()
