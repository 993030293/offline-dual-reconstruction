"""Validate real SIFT artifacts and cache failure lifecycle in isolated fixtures."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
sys.path.insert(0, str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002"))
from runtime import atomic_json, data_identity
from feature_localization import FeatureBank
import query_feature_cache as implementation
from query_feature_cache import QueryFeatureCache, option_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(8)
    options = pycolmap.FeatureExtractionOptions()
    options.sift.max_num_features, options.sift.peak_threshold = 12000, .004
    extractor = pycolmap.Sift(options, pycolmap.Device.cpu)
    image = np.random.default_rng(20261005).integers(0, 256, (240, 320), dtype=np.uint8)
    source = args.output/"source.png"
    if not cv2.imwrite(str(source), image):
        raise RuntimeError("Lifecycle fixture could not be encoded")

    def bank(name, selected=extractor):
        directory = args.output/name
        directory.mkdir()
        return SimpleNamespace(cache=directory, extractor=selected,
            protocol={"SIFT_max_features": 12000, "SIFT_peak_threshold": .004})

    def rejected(call, kind):
        try:
            call()
        except kind:
            return True
        return False

    checks = {}
    oracle_bank = bank("oracle")
    oracle = FeatureBank.extra(oracle_bank, source)
    cache_root = args.output/"cache"
    cache = QueryFeatureCache(cache_root, FeatureBank.extra)
    cold_bank, warm_bank = bank("cold"), bank("warm")
    cold, warm = cache(cold_bank, source), cache(warm_bank, source)
    checks["real_SIFT_cold_warm_arrays_and_bytes_exact"] = (
        all(np.array_equal(a, b) and np.array_equal(a, c) for a, b, c in zip(oracle, cold, warm))
        and (oracle_bank.cache/"source.npz").read_bytes() == (warm_bank.cache/"source.npz").read_bytes())
    checks["one_actual_cold_and_one_warm_hit"] = cache.misses == 1 and cache.hits == 1
    warm[0][:] = 0
    reopened = QueryFeatureCache(cache_root, FeatureBank.extra)
    repeated = reopened(bank("reopened"), source)
    checks["returned_mutation_cannot_change_immutable_artifact"] = all(np.array_equal(a, b) for a, b in zip(oracle, repeated))
    checks["preexisting_output_rejected"] = rejected(lambda: cache(cold_bank, source), FileExistsError)
    altered = image.copy()
    altered[80:140, 80:140] ^= 255
    if not cv2.imwrite(str(source), altered):
        raise RuntimeError("Changed lifecycle fixture could not be encoded")
    cache(bank("changed_image"), source)
    checks["same_filename_changed_raw_image_misses"] = cache.misses == 2
    if not cv2.imwrite(str(source), image):
        raise RuntimeError("Original lifecycle fixture could not be restored")
    bad_protocol = bank("bad_protocol")
    bad_protocol.protocol["SIFT_peak_threshold"] = .008
    checks["changed_protocol_rejected"] = rejected(lambda: cache(bad_protocol, source), ValueError)
    changed_options = pycolmap.FeatureExtractionOptions()
    changed_options.sift.max_num_features, changed_options.sift.peak_threshold = 12000, .004
    changed_options.sift.edge_threshold = 11
    changed_extractor = pycolmap.Sift(changed_options, pycolmap.Device.cpu)
    checks["unrecorded_actual_extractor_option_rejected"] = rejected(lambda: cache(bank("bad_options", changed_extractor), source), ValueError)
    cv2.setNumThreads(1)
    checks["changed_runtime_threads_rejected"] = rejected(lambda: cache(bank("bad_threads"), source), ValueError)
    cv2.setNumThreads(8)
    original_identity = implementation.file_identity
    helper = Path(sys.modules["feature_localization"].__file__).resolve()

    def changed_identity(path):
        value = original_identity(path)
        if Path(path).resolve() == helper:
            value["sha256"] = "0"*64
        return value

    with patch.object(implementation, "file_identity", changed_identity):
        different_source = QueryFeatureCache(cache_root, FeatureBank.extra)
        value = different_source(bank("changed_source"), source)
        checks["changed_feature_producer_source_misses"] = different_source.misses == 1 and all(np.array_equal(a, b) for a, b in zip(oracle, value))
    checks["unsupported_option_type_rejected"] = rejected(lambda: option_identity(object()), TypeError)
    checks["cold_budget_failure_unpublished"] = rejected(
        lambda: QueryFeatureCache(args.output/"tiny_cold", FeatureBank.extra, maximum_bytes=1)(bank("tiny_cold_output"), source), MemoryError)
    checks["warm_budget_failure_rejected"] = rejected(
        lambda: QueryFeatureCache(cache_root, FeatureBank.extra, maximum_bytes=1)(bank("tiny_warm_output"), source), MemoryError)
    entry = cache_root/cache.records[0]["key"]
    for kind in ("corrupt_payload", "metadata_mismatch", "incomplete"):
        fixture = args.output/kind
        shutil.copytree(cache_root, fixture)
        copied = fixture/entry.name
        if kind == "corrupt_payload":
            payload = copied/"features.npz"
            value = bytearray(payload.read_bytes())
            value[-16] ^= 1
            payload.write_bytes(value)
            exception = ValueError
        elif kind == "metadata_mismatch":
            metadata = copied/"ENTRY.json"
            value = json.loads(metadata.read_text(encoding="utf-8"))
            value["key_input"]["schema"] = 999
            metadata.write_text(json.dumps(value), encoding="utf-8")
            exception = ValueError
        else:
            (copied/"ENTRY.json").unlink()
            exception = RuntimeError
        checks[kind+"_rejected_without_silent_reextraction"] = rejected(
            lambda: QueryFeatureCache(fixture, FeatureBank.extra)(bank(kind+"_output"), source), exception)
    (cache_root/"unpublished.tmp").write_bytes(b"unpublished")
    value = QueryFeatureCache(cache_root, FeatureBank.extra)(bank("ignore_temporary"), source)
    checks["uncommitted_temporary_file_ignored"] = all(np.array_equal(a, b) for a, b in zip(oracle, value))
    result = {"status": "real_SIFT_lifecycle_passed" if len(checks) == 16 and all(checks.values()) else "failed",
        "checks": checks, "sources": [data_identity(Path(__file__)), data_identity(Path(implementation.__file__))],
        "fixture_scope": "cache_lifecycle_only; performance_is_from_separate_real24_query_entries", "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] == "failed":
        raise AssertionError("Query cache lifecycle validation failed")


if __name__ == "__main__":
    main()
