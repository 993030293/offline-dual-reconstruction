"""Exercise cold and warm artifacts on complete inputs from this raw campaign."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3), str(ROOT / "02_code/box2_lio_guided_sfm_v1_20261002")]
from runtime import atomic_json, data_identity
import run_initial as initial
from encoded_jpeg_cache import EncodedJpegCache
from calibrated_reference_cache import CalibratedReferenceCache
from query_feature_cache import QueryFeatureCache
from feature_localization import FeatureBank
from exact_bayer_gray import raw_gray_quality_execution
from planar_corner import planar_corner_execution
from vector_gray_stats import vector_gray_stats_execution
from quality_histogram import histogram_quality_execution


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=("quality", "jpeg", "reference", "query"), default=["quality", "jpeg", "reference", "query"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    raw_identity = {"path": str(args.raw.resolve()), "bytes": args.raw.stat().st_size, "mtime_ns": args.raw.stat().st_mtime_ns}
    gold = [json.loads(line) for line in (args.side / "initial/frame_decisions.jsonl").read_text().splitlines()]
    index = initial.McapIndex(args.raw)
    total = index.message_count(initial.baseline.CAMERA_TOPIC)
    if total != len(gold) or [row["camera_frame"] for row in gold] != list(range(total)):
        raise ValueError("A complete raw score denominator is required")
    result = {"status": "running", "source": data_identity(Path(__file__)), "raw": raw_identity,
        "source_initial_contract": data_identity(args.side / "initial/CONTRACT.json"), "modules": {},
        "raw_frame_count": total, "old_assets_used": False, "same_campaign_assets_are_diagnostic_oracles": True,
        "OS_page_cache_cleared": False, "speed_measurements_may_overlap_campaign_CPU_and_IO": True,
        "cold_construction_and_warm_reuse_costs_are_separate": True, "whole_pipeline_speed_claim": False}
    atomic_json(args.output / "CONTRACT.json", result)

    def save():
        atomic_json(args.output / "RESULT.json", result)

    def measure(function):
        before = time.monotonic()
        value = function()
        return value, time.monotonic()-before

    def messages(numbers):
        return initial.iter_single_copy_messages(index, initial.baseline.CAMERA_TOPIC, numbers)

    for family in args.families:
        family_started = time.monotonic()
        module_key = {"quality": "P1_PROVENANCE_CACHE", "jpeg": "Q33", "reference": "Q47", "query": "Q48"}[family]
        try:
            if family == "quality":
                cache = initial.quality_engine.ScoreCache(args.output / "quality_cache", {
                    "scorer": data_identity(Path(initial.quality_engine.__file__)), "OpenCV": cv2.__version__,
                    "raw": raw_identity, "pose_and_selection_cached": False})
                trials, maximum_delta = [], 0.
                for run in ("cold", "warm1", "warm2"):
                    started = time.monotonic()
                    traces = {}
                    with ExitStack() as stack:
                        libs = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
                        traces["R149"] = stack.enter_context(planar_corner_execution(libs / "qc001/b0/native_planar_corner.dll"))
                        traces["R144"] = stack.enter_context(vector_gray_stats_execution(libs / "vector_gray_stats_001/b0/native_vector_gray_stats.dll"))
                        traces["R147_R141"] = stack.enter_context(raw_gray_quality_execution(libs / "bg001/b0/native_bayer_gray.dll", initial.demosaic_rggb8_to_bgr))
                        traces["R104_builder"] = stack.enter_context(histogram_quality_execution(True))
                        rows, trace = initial.quality_engine.score_messages(messages(range(total)), initial.decode_bayer,
                            lambda value: value, initial.quality_engine.compact_quality, workers=4, capacity=16,
                            cache=cache, maximum_seconds=1200)
                    for expected, actual in zip(gold, rows):
                        if actual["camera_header_ns"] != expected["camera_header_ns"]:
                            raise ValueError("A cached raw frame clock changed")
                        for key in initial.quality_engine.FIELDS:
                            delta = abs(actual["quality"][key]-expected["quality"][key])
                            maximum_delta = max(maximum_delta, delta)
                            if delta > 1e-8 or (key in ("corner_count_half_resolution", "occupied_grid_cells_4x4") and delta != 0):
                                raise ValueError("The complete original raw score oracle changed")
                    if len(rows) != total or trace["cache_hits"] != (0 if run == "cold" else total):
                        raise ValueError("Raw cache hit coverage differs from its complete cold/warm contract")
                    trials.append({"run": run, "seconds": time.monotonic()-started, "trace": trace, "native_trace": traces})
                    result["modules"]["P1_PROVENANCE_CACHE"] = {"status": "running", "trials": trials}
                    save()
                result["modules"]["P1_PROVENANCE_CACHE"] = {"status": "complete_score_oracle_verified_cold_and_warm",
                    "trials": trials, "maximum_absolute_score_delta": maximum_delta, "complete_rows_per_run": total,
                    "warm_benefit_relative_to_uncached_optimized_requires_same_resource_timing": True}
            elif family == "jpeg":
                manifest = read(args.side / "initial/INITIAL_FRAME_MANIFEST.json")
                evaluation = read(args.side / "initial/EVALUATION_FRAMES.json")["rows"]
                expected = {row["camera_frame"]: args.side / "initial/images" / row["image_name"] for row in manifest["selected_frames"]}
                expected.update({row["camera_frame"]: args.side / "initial/evaluation_images" / row["image_name"] for row in evaluation})
                cache = EncodedJpegCache(args.output / "jpeg_cache")
                trials = []
                try:
                    for run in ("native", "cold", "warm1", "warm2"):
                        started = time.monotonic()
                        for frame, message in messages(sorted(expected)):
                            _, _, raw = initial.decode_bayer(message["payload"])
                            image = initial.demosaic_rggb8_to_bgr(raw)
                            encode = cv2.imencode if run == "native" else cache.encode
                            ok, encoded = encode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, manifest["policy"]["jpeg_quality"]])
                            if not ok or encoded.tobytes() != expected[frame].read_bytes():
                                raise ValueError("A complete initial or evaluation JPEG differs from its fresh raw native bytes")
                        trials.append({"run": run, "seconds": time.monotonic()-started, "complete_images": len(expected), "cache": cache.receipt()})
                    if cache.hits != 2*len(expected) or cache.misses != len(expected):
                        raise ValueError("The complete JPEG hit/miss coverage changed")
                    result["modules"]["Q33"] = {"status": "complete_native_JPEG_bytes_preserved_cold_and_warm", "trials": trials,
                        "cold_overhead_seconds": trials[1]["seconds"]-trials[0]["seconds"],
                        "warm_reduction_fraction": 1-np.median([row["seconds"] for row in trials[2:]])/trials[0]["seconds"]}
                finally:
                    cache.close()
            elif family == "reference":
                admission = read(args.side / "export/TRAINING_ADMISSION.json")
                model = pycolmap.Reconstruction(Path(admission["training_opencv_model"]))
                camera = next(iter(model.cameras.values()))
                native_undistort, native_save = pycolmap.undistort_image, Image.Image.save
                cache = CalibratedReferenceCache(args.output / "reference_cache", native_undistort, native_save)
                trials = []
                try:
                    for run in ("native", "cold", "warm1", "warm2"):
                        folder = args.output / ("reference_"+run)
                        folder.mkdir()
                        started = time.monotonic()
                        for row in read(args.side / "initial/EVALUATION_FRAMES.json")["rows"]:
                            image = args.side / "initial/evaluation_images" / row["image_name"]
                            with Image.open(image) as photo:
                                bitmap = pycolmap.Bitmap.from_array(np.asarray(photo.convert("RGB")))
                            undistort = native_undistort if run == "native" else cache.undistort
                            output, calibrated = undistort(pycolmap.UndistortCameraOptions(), bitmap, camera)
                            target = folder / (image.stem+".png")
                            (native_save if run == "native" else cache.save)(Image.fromarray(output.to_array()), target)
                            golden = args.side / "localization/reference_images" / target.name
                            if target.read_bytes() != golden.read_bytes():
                                raise ValueError("A native calibrated PNG differs from the complete frozen evaluation oracle")
                        trials.append({"run": run, "seconds": time.monotonic()-started, "complete_images": 24})
                    result["modules"]["Q47"] = {"status": "all24_native_reference_PNG_bytes_preserved", "trials": trials,
                        "cache": cache.receipt(), "warm_reduction_fraction": 1-np.median([r["seconds"] for r in trials[2:]])/trials[0]["seconds"]}
                finally:
                    cache.close()
            elif family == "query":
                cv2.setNumThreads(8)
                options = pycolmap.FeatureExtractionOptions()
                options.sift.max_num_features, options.sift.peak_threshold = 12000, .004
                cache = QueryFeatureCache(args.output / "query_cache", FeatureBank.extra)
                trials = []
                for run in ("native", "cold", "warm1", "warm2"):
                    bank = FeatureBank.__new__(FeatureBank)
                    bank.protocol = {"SIFT_max_features": 12000, "SIFT_peak_threshold": .004}
                    bank.extractor = pycolmap.Sift(options, pycolmap.Device.cpu)
                    bank.cache = args.output / ("query_"+run)
                    bank.cache.mkdir()
                    started = time.monotonic()
                    for row in read(args.side / "initial/EVALUATION_FRAMES.json")["rows"]:
                        image = args.side / "initial/evaluation_images" / row["image_name"]
                        actual = FeatureBank.extra(bank, image) if run == "native" else cache(bank, image)
                        with np.load(args.side / "localization/feature_cache" / (image.stem+".npz")) as saved:
                            if not all(np.array_equal(value, saved[name]) for value, name in zip(actual, ("keypoints", "descriptors"))):
                                raise ValueError("A native CPU query feature array differs from its complete fresh extraction")
                    trials.append({"run": run, "seconds": time.monotonic()-started, "complete_images": 24})
                result["modules"]["Q48"] = {"status": "all24_native_query_arrays_preserved", "trials": trials, "cache": cache.receipt(),
                    "warm_reduction_fraction": 1-np.median([r["seconds"] for r in trials[2:]])/trials[0]["seconds"]}
        except BaseException as error:
            result["modules"][family] = {"status": "failed_complete_cache_oracle", "error": repr(error), "partial_outputs_preserved": True}
        result["modules"].setdefault(module_key, {"status": "failed_complete_cache_oracle"})["complete_family_wall_seconds"] = time.monotonic()-family_started
        save()
    if raw_identity != {"path": str(args.raw.resolve()), "bytes": args.raw.stat().st_size, "mtime_ns": args.raw.stat().st_mtime_ns}:
        raise ValueError("The raw recording changed during read-only cache diagnostics")
    result["status"] = "all_requested_cache_families_resolved_with_failures_retained"
    save()
    print(json.dumps({"status": result["status"], "modules": {key: row["status"] for key, row in result["modules"].items()}}))


if __name__ == "__main__":
    main()
