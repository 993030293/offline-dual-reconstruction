"""Audit the complete fixed matching input against native cache dependencies."""

import collections
import json
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import psutil

from audit_native_match_sources_013 import BASE, PREFIX, ROOT, save
from runtime import data_identity


def main():
    output = BASE / "a0"
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    source_root = BASE / "s1/sources"
    old = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"
    source_paths = [Path(__file__), source_root / "exe/feature.cc", source_root / "controllers/pairing.h",
        source_root / "controllers/pairing.cc", source_root / "controllers/feature_matching.cc",
        source_root / "controllers/feature_matching_utils.cc", source_root / "util/cache.h",
        BASE.parent / "oup010/p0/sources/sift.cc"]
    inputs = [data_identity(old / name) for name in ("features.db", "guided_pairs.txt", "STATE.json")]
    sources = [data_identity(path) for path in source_paths]
    save(output / "PROTOCOL.json", {"sources": sources, "inputs": inputs, "fixed_commit": "a0d785f",
        "parent_count": 36, "new_optimization_count": 0, "read_only_original_inputs": True,
        "source_inference_not_native_cache_or_GPU_trace": True})
    fetched = []
    for name in ("matcher_cache.h", "matcher_cache.cc"):
        path = output / name
        url = PREFIX + "controllers/" + name
        stamp = time.perf_counter()
        with urllib.request.urlopen(url, timeout=12) as response:
            data = response.read(200001)
        content = data.decode("utf-8")
        if len(data) > 200000 or "Copyright (c), ETH Zurich and UNC Chapel Hill." not in content or "Redistribution and use in source and binary forms" not in content:
            raise AssertionError("Pinned original cache sources and license are required")
        with path.open("xb") as handle:
            handle.write(data)
        fetched.append({"url": url, "source": data_identity(path), "seconds": time.perf_counter() - stamp})
        save(output / ("FETCH_" + name + ".json"), fetched[-1])
    documents = {path.name: path.read_text(encoding="utf-8") for path in source_paths[1:]}
    cache_source = (output / "matcher_cache.cc").read_text(encoding="utf-8")
    imported = documents["pairing.h"].split("struct ImportedPairingOptions {")[1].split("};")[0]
    block_size = int(re.search(r"int block_size = (\d+);", imported).group(1))
    with sqlite3.connect((old / "features.db").as_uri() + "?mode=ro", uri=True) as database:
        names = dict(database.execute("SELECT name,image_id FROM images"))
        counts = dict(database.execute("SELECT image_id,rows FROM keypoints"))
        descriptors = dict(database.execute("SELECT image_id,cols FROM descriptors"))
    lines = (old / "guided_pairs.txt").read_text(encoding="utf-8").splitlines()
    pairs = [tuple(names[name] for name in line.split()) for line in lines]
    canonical = [tuple(sorted(pair)) for pair in pairs]
    changes = [sum(index == 0 or pair[side] != pairs[index - 1][side]
        for index, pair in enumerate(pairs)) for side in (0, 1)]
    descriptor_bytes = [sum(counts[pair[side]] * descriptors[pair[side]]
        for index, pair in enumerate(pairs) if index == 0 or pair[side] != pairs[index - 1][side]) for side in (0, 1)]
    slot_lower_bound = len(pairs) + 1
    pairing_body = documents["pairing.cc"].split("ImportedPairGenerator::ImportedPairGenerator(", 1)[1]
    pairing_body = pairing_body.split("ExistingMatchedPairGenerator::", 1)[0]
    cache_capacity_safe = "inline size_t CacheSize() const { return block_size; }" in imported and len(names) <= block_size
    log = (BASE.parent / "oup009/R165/c1/b/worker0.log").read_text(encoding="utf-8")
    block_numbers = re.findall(r"Processing block \[(\d+)/(\d+)\]", log)
    stage = next(row for row in json.loads((old / "STATE.json").read_text(encoding="utf-8"))["stages"] if row["name"] == "guided_matching")
    command = stage["command"]
    checks = {
        "original_complete1002_1779059_and11583_unique_nonself_pairs": len(names) == 1002 and sum(counts.values()) == 1779059 and len(pairs) == len(set(canonical)) == 11583 and all(a != b for a, b in pairs),
        "actual_original_match_type_pairs_not_feature_raw_import": command[command.index("--match_type") + 1] == "pairs" and 'if (match_type == "pairs")' in documents["feature.cc"] and "ImportedPairingOptions pairing_options;" in documents["feature.cc"],
        "native_imported_default_block_and_image_cache_capacity1225": block_size == 1225 and cache_capacity_safe,
        "actual_saved_native_log_all_ten1225_pair_blocks": len(block_numbers) == 10 and [int(a) for a, _ in block_numbers] == list(range(1, 11)) and all(b == "10" for _, b in block_numbers),
        "descriptor_keypoint_and_existence_cache_size_and_locked_database_load": all(name in cache_source for name in ("keypoints_cache_", "descriptors_cache_", "keypoints_exists_cache_", "descriptors_exists_cache_")) and "cache_size_" in cache_source and "database_mutex_" in cache_source and "Clear()" not in cache_source,
        "native_one_shared_future_per_cache_entry_and_no_capacity_eviction_below_limit": "shared_future" in documents["cache.h"] and "if (elems_map_.size() > max_num_elems_)" in documents["cache.h"],
        "imported_next_preserves_input_order_with_no_hidden_reorder": "ReadImagePairsText" in pairing_body and "block_image_pairs_.push_back(image_pairs_[j]);" in pairing_body and "sort(" not in pairing_body,
        "native_gpu_matcher_already_caches_two_previous_descriptor_slots": all(token in documents["sift.cc"] for token in ("prev_image_id1_ != image1.image_id", "prev_image_id2_ != image2.image_id", "sift_match_gpu_.SetDescriptors(")),
        "original_source_image_groups_already_contiguous_and_minimum_side1_loads": changes[0] == len({pair[0] for pair in pairs}) == 1001,
        "remaining_GPU_upload_count_opportunity_only_source_inference_under8percent": slot_lower_bound <= sum(changes) and (sum(changes) - slot_lower_bound) / sum(changes) < .08,
        "all_original_source_and_input_data_identities_unchanged": all(data_identity(row["path"]) == row for row in sources + inputs),
    }
    result = {"status": "host_cache_locality_lead_source_rejected0_GPU_upload_timing_unverified",
        "checks": checks, "sources": sources, "inputs": inputs, "fetched": fetched,
        "input": {"images": len(names), "features": sum(counts.values()), "pairs": len(pairs)},
        "native_host_cache_capacity": block_size,
        "host_cache_eviction_not_possible_for1002_keys_at_capacity1225_from_source": cache_capacity_safe,
        "GPU_slot_upload_changes_source_inference_not_actual_trace": changes,
        "GPU_descriptor_input_bytes_source_inference_not_measured_transfer": descriptor_bytes,
        "minimum_slot_upload_count_for_unique_ordered_pairs_no_endpoint_reversal": slot_lower_bound,
        "optimistic_upload_count_reduction_upper_not_time_or_byte_bound": (sum(changes) - slot_lower_bound) / sum(changes),
        "current_complete_matching_native_log_block_counts": block_numbers,
        "no_algorithm_native_matching_quality_or_speed_experiment": True,
        "no_new_R168_registered_or_implemented": True,
        "cost_known_s1_guard_seconds": json.loads((BASE / "s1_GUARD.json").read_text())["elapsed_seconds"],
        "prior_failed_guard_launch_seconds": .1881376,
        "elapsed_seconds": time.perf_counter() - started,
        "writer_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "parent_count": 36, "new_optimization_count": 0}
    save(output / "RESULT.json", result)
    print(json.dumps({"checks": checks, "input": result["input"], "cache_capacity": block_size,
        "GPU_slot_changes_inference": changes, "upload_count_upper_not_time": result["optimistic_upload_count_reduction_upper_not_time_or_byte_bound"], "new_count": 0}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The complete cache source audit failed verification")


if __name__ == "__main__":
    main()
