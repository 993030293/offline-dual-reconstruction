"""Exercise real encoder bytes, cache invalidation, ownership and failure cleanup."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

from encoded_jpeg_cache import EncodedJpegCache, cached_jpeg_encoding
V1 = Path(__file__).resolve().parents[2]/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    checks = {}
    original = cv2.imencode
    image = np.random.default_rng(0).integers(0, 256, (67, 91, 3), dtype=np.uint8)
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 97]
    cache_dir = args.output/"cache"
    with cached_jpeg_encoding(cache_dir) as cache:
        oracle = original(".jpg", image, parameters)[1]
        cold = cv2.imencode(".jpg", image, parameters)[1]
        hot = cv2.imencode(".jpg", image.copy(), parameters)[1]
        checks["cold_hot_bytes_shape_dtype_exact"] = (np.array_equal(oracle, cold) and np.array_equal(oracle, hot)
            and oracle.shape == hot.shape and oracle.dtype == hot.dtype and hot.flags.writeable)
        checks["actual_cold_and_hit"] = cache.misses == 1 and cache.hits == 1
        noncontiguous = image[::-1, ::2]
        expected = original(".jpg", noncontiguous, parameters)[1]
        value = cv2.imencode(".jpg", noncontiguous, parameters)[1]
        checks["noncontiguous_encoder_exact"] = np.array_equal(expected, value)
        changed = image.copy()
        changed[20, 20] ^= 255
        cv2.imencode(".jpg", changed, parameters)
        cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 89])
        checks["pixels_shape_and_parameters_invalidate"] = cache.misses == 4 and cache.hits == 1
        try:
            EncodedJpegCache(cache_dir)
        except FileExistsError:
            checks["concurrent_owner_rejected"] = True
        else:
            checks["concurrent_owner_rejected"] = False
        try:
            cv2.imencode(".png", image)
        except ValueError:
            checks["unsupported_encoder_rejected"] = True
        try:
            cv2.imencode(".jpg", image.astype(np.float32))
        except ValueError:
            checks["implicit_pixel_conversion_rejected"] = True
    checks["function_restored_and_owner_released"] = cv2.imencode is original and not (cache_dir/"OWNER.json").exists()
    (cache_dir/"interrupted.tmp").write_bytes(b"partial unpublished JPEG")
    with patch.object(cv2, "getBuildInformation", return_value="different encoder build"):
        with cached_jpeg_encoding(cache_dir) as changed_build:
            new = cv2.imencode(".jpg", image, parameters)[1]
            checks["build_change_misses_and_preserves_pixels"] = changed_build.misses == 1 and np.array_equal(new, oracle)
    with cached_jpeg_encoding(cache_dir) as reopened:
        value = cv2.imencode(".jpg", image, parameters)[1]
        checks["reopen_hit_and_uncommitted_file_ignored"] = reopened.hits == 1 and np.array_equal(value, oracle)
    try:
        with cached_jpeg_encoding(args.output/"tiny", maximum_bytes=10):
            cv2.imencode(".jpg", image, parameters)
    except MemoryError:
        checks["byte_budget_rejected_and_restored"] = cv2.imencode is original and not (args.output/"tiny/OWNER.json").exists()
    path = next(cache_dir.glob("*.jpgc"))
    damaged = bytearray(path.read_bytes())
    damaged[-8] ^= 1
    path.write_bytes(damaged)
    corruption_rejected = False
    try:
        with cached_jpeg_encoding(cache_dir):
            for source, flags in ((image, parameters), (noncontiguous, parameters), (changed, parameters),
                                  (image, [cv2.IMWRITE_JPEG_QUALITY, 89])):
                cv2.imencode(".jpg", source, flags)
    except ValueError:
        corruption_rejected = True
    if not corruption_rejected:
        with patch.object(cv2, "getBuildInformation", return_value="different encoder build"):
            try:
                with cached_jpeg_encoding(cache_dir):
                    cv2.imencode(".jpg", image, parameters)
            except ValueError:
                corruption_rejected = True
    checks["committed_corruption_rejected_no_silent_reencode"] = corruption_rejected
    checks["failure_restores_encoder_and_releases_owner"] = cv2.imencode is original and not (cache_dir/"OWNER.json").exists()
    with patch.object(cv2, "imencode", return_value=(False, None)):
        cache = EncodedJpegCache(args.output/"failed_encoder")
        try:
            cache.encode(".jpg", image, parameters)
        except RuntimeError:
            checks["native_encoder_failure_propagates"] = not list(cache.directory.glob("*.jpgc"))
        finally:
            cache.close()
    result = {"status": "numeric_lifecycle_passed" if len(checks) == 14 and all(checks.values()) else "failed",
        "checks": checks, "counted": False, "sources": [data_identity(Path(__file__)),
            data_identity(Path(__file__).with_name("encoded_jpeg_cache.py"))]}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result))
    if result["status"] == "failed":
        raise AssertionError("JPEG cache numerical/lifecycle validation failed")


if __name__ == "__main__":
    main()
