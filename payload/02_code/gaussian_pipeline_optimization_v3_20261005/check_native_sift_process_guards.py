"""Verify native cache bytes, ordered errors and complete process shutdown."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from query_feature_cache import QueryFeatureCache, file_identity
from native_sift_process import NativeSiftProcess, native_sift_process, options
import feature_localization


def bank(folder):
    folder.mkdir(parents=True, exist_ok=False)
    value = feature_localization.FeatureBank.__new__(feature_localization.FeatureBank)
    value.protocol = {"SIFT_max_features": 12000, "SIFT_peak_threshold": .004}
    value.extractor = pycolmap.Sift(options(), pycolmap.Device.cpu)
    value.cache = folder
    return value


def arrays_equal(left, right):
    return len(left) == len(right) and all(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        for a, b in zip(left, right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = [Path(__file__), Path(__file__).with_name("native_sift_process.py"), Path(feature_localization.__file__)]
    identities = [data_identity(p) for p in sources]
    original_extra, original_setattr = feature_localization.FeatureBank.extra, feature_localization.FeatureBank.__dict__.get("__setattr__")
    cv2.setNumThreads(8)
    generator = np.random.default_rng(20261006)
    paths = []
    for ordinal, (width, height, kind) in enumerate(((91, 73, "constant"), (320, 257, "thin_curve"),
            (640, 360, "random"), (257, 511, "translated"))):
        pixels = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
        if kind == "constant":
            pixels[:] = 127
        elif kind == "thin_curve":
            pixels[:] = 0
            x = np.arange(width)
            y = np.rint(height*.5+height*.3*np.sin(x*.047)).astype(int)
            pixels[y, x] = 255
        elif kind == "translated":
            pixels = np.roll(pixels, (7, -13), (0, 1))
        path = args.output/f"image{ordinal}.png"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    records, checks = [], {}
    baseline_bank = bank(args.output/"b")
    original_cache = QueryFeatureCache(args.output/"bc", original_extra)
    expected = [original_cache(baseline_bank, path) for path in paths]
    with native_sift_process(paths) as trace:
        current_bank = bank(args.output/"c")
        current_cache = QueryFeatureCache(args.output/"cc", feature_localization.FeatureBank.extra)
        current = [current_cache(current_bank, path) for path in paths]
        checks["original_extra_pointer_and_extractor_identity_exact"] = feature_localization.FeatureBank.extra is original_extra
        checks["original_extractor_metadata_and_cache_key_exact"] = original_cache.extractor_identity == current_cache.extractor_identity
        checks["all4_native_arrays_exact"] = all(arrays_equal(a, b) for a, b in zip(expected, current))
        checks["all4_original_feature_NPZ_bytes_exact"] = all(file_identity(baseline_bank.cache/(path.stem+".npz"))
            == file_identity(current_bank.cache/(path.stem+".npz")) for path in paths)
        checks["all4_committed_cache_metadata_exact"] = sorted(p.name for p in original_cache.root.iterdir()) == sorted(p.name for p in current_cache.root.iterdir())
        for folder in original_cache.root.iterdir():
            checks["all4_committed_cache_metadata_exact"] &= json.loads((folder/"ENTRY.json").read_text()) == json.loads((current_cache.root/folder.name/"ENTRY.json").read_text())
    records.append(trace)
    checks["binding_restored_after_normal_close"] = trace["binding_restored"] and trace["broker"]["closed"]
    with native_sift_process(paths, enabled=False) as warm_trace:
        warm_bank = bank(args.output/"w")
        warm = [current_cache(warm_bank, path) for path in paths]
    checks["warm_hits_use_original_route_and_bytes"] = warm_trace["broker"] is None and all(arrays_equal(a, b) for a, b in zip(expected, warm))
    records.append(warm_trace)
    path = paths[0]
    duplicate_bank = bank(args.output/"dup")
    duplicate_cache = QueryFeatureCache(args.output/"dc", original_extra)
    duplicate_cache(duplicate_bank, path)
    try:
        duplicate_cache(duplicate_bank, path)
        checks["duplicate_original_target_error"] = False
    except FileExistsError:
        checks["duplicate_original_target_error"] = True
    broker = NativeSiftProcess([paths[0]])
    selected = options()
    selected.sift.peak_threshold = .01
    try:
        broker.bind(pycolmap.Sift(selected, pycolmap.Device.cpu))
        checks["changed_options_rejected_before_pool"] = False
    except ValueError:
        checks["changed_options_rejected_before_pool"] = broker.pool is None
    finally:
        broker.close()
    records.append(broker.trace)
    bad = args.output/"invalid.png"
    bad.write_bytes(b"This is deliberately not an image.")
    late = NativeSiftProcess([paths[1], bad])
    try:
        proxy = late.bind(pycolmap.Sift(options(), pycolmap.Device.cpu))
        first = proxy.extract(cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE))
        checks["later_worker_error_does_not_skip_first_result"] = arrays_equal(first, expected[1])
        try:
            proxy.extract(np.zeros((3, 3), np.uint8))
            checks["original_image_decode_ValueError_in_order"] = False
        except ValueError as error:
            checks["original_image_decode_ValueError_in_order"] = str(error) == f"Image decode failed: {bad.resolve()}"
    finally:
        late.close()
    records.append(late.trace)
    try:
        with native_sift_process(paths) as abort_trace:
            abort_bank = bank(args.output/"abort")
            abort_bank.extractor.extract(cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE))
            raise RuntimeError("intentional caller abort")
    except RuntimeError as error:
        checks["caller_abort_preserved_and_workers_joined"] = str(error) == "intentional caller abort" and abort_trace["broker"]["closed"]
    records.append(abort_trace)
    try:
        late.extract(np.zeros((3, 3), np.uint8))
        checks["closed_broker_cannot_be_reused"] = False
    except RuntimeError:
        checks["closed_broker_cannot_be_reused"] = True
    checks["all_sources_unchanged"] = identities == [data_identity(p) for p in sources]
    checks["original_callbacks_restored"] = feature_localization.FeatureBank.extra is original_extra and feature_localization.FeatureBank.__dict__.get("__setattr__") is original_setattr
    checks["all_created_pools_closed"] = all(r.get("broker", r).get("closed", True) for r in records if r.get("broker", r) is not None)
    atomic_json(args.output/"RESULT.json", {"status": "passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "records": records, "sources": identities, "whole_seconds": time.perf_counter()-started,
        "shape_fixture_count": len(paths), "counted": False, "formal_cost_and_external": "not_run"})
    print(json.dumps({"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks, "counted": False}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
