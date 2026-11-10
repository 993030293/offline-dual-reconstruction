"""Compare full native checkpoint loading, validation and cleanup with private mapping."""

import argparse
import gc
import json
import mmap
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from run_codec_campaign import exact


def device_layout(state):
    import torch
    rows, groups = [], {}

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            token = (str(value.device), storage.data_ptr(), storage.nbytes())
            if token not in groups:
                groups[token] = len(groups)
            rows.append((path, list(value.shape), list(value.stride()), str(value.dtype), str(value.device), value.storage_offset(), groups[token]))
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, path+"/"+str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, path+"/"+str(index))

    walk(state, "state")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoint) != 3 or len(set(args.checkpoint)) != 3:
        raise ValueError("Three distinct native complete checkpoint inputs are required")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    if torch.__version__ != "2.7.1+cu128" or torch.serialization.get_default_mmap_options() != mmap.MAP_PRIVATE:
        raise ValueError("Select the original loader before execution outside the fixed private-map contract")
    paths = [Path(__file__), Path(__file__).with_name("run_codec_campaign.py"), *args.checkpoint]
    identity = [data_identity(p) for p in paths]
    source_ids = {Path(r["path"]): r for r in identity[2:]}
    atomic_json(args.output/"PROTOCOL.json", {"sources_inputs": identity, "three_real_states_each_three_full_pairs": True,
        "default_saved_CPU_and_CUDA_device_labels_unchanged": True, "mmap_options": "MAP_PRIVATE",
        "ZIP_format_checked_before_execution": True, "opaque_complete_native_state_not_object_specific": True,
        "full_SHA_before_after_two_loads_full_value_layout_and_device_verify_gc_and_CUDA_cleanup_in_phase": True,
        "values_compared_to_independent_unmodified_native_oracle": True,
        "minimum_sum_source_median_stage_reduction": .1, "maximum_CUDA_reserved_bytes": 4*1024**3,
        "loader_only_not_restore_or_new_training": True, "real_native_recovery_required_before_acceptance": True,
        "old_largest_checkpoint_C212_iteration8000_preserved": True, "counted": False})
    setup = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    common_setup = time.perf_counter()-setup
    pairs = []
    try:
        for index, path in enumerate(args.checkpoint):
            layouts = {}
            oracle_started = time.perf_counter()
            oracle = torch.load(path, weights_only=False)
            torch.cuda.synchronize()
            oracle_layout = device_layout(oracle)
            oracle_seconds = time.perf_counter()-oracle_started
            for trial in range(3):
                times, guards, peaks = {}, {}, {}
                order = ("native", "mapped") if trial % 2 == 0 else ("mapped", "native")
                for mode in order:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    before = data_identity(path)
                    if before != source_ids[path]:
                        raise ValueError("The original complete checkpoint input changed")
                    with path.open("rb") as stream:
                        supported = torch.serialization._is_zipfile(stream)
                    if not supported:
                        raise ValueError("Preselect the original native loader for a non-ZIP checkpoint")
                    first = torch.load(path, weights_only=False, mmap=(mode == "mapped"))
                    independent = torch.load(path, weights_only=False, mmap=(mode == "mapped"))
                    torch.cuda.synchronize()
                    if not isinstance(first, dict) or first.get("schema_version") != 1 or "gaussian_capture" not in first:
                        raise ValueError("The trusted complete native training state schema changed")
                    layout = device_layout(first)
                    guards[mode] = exact(first, independent) and layout == device_layout(independent)
                    guards[mode] = guards[mode] and exact(first, oracle) and layout == oracle_layout
                    if path in layouts:
                        guards[mode] = guards[mode] and layouts[path] == layout
                    else:
                        layouts[path] = layout
                    payload_digest = []
                    for value in first["gaussian_capture"]:
                        if isinstance(value, torch.Tensor):
                            payload_digest.append((list(value.shape), str(value.dtype), str(value.device)))
                    del first, independent
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    after = data_identity(path)
                    times[mode] = time.perf_counter()-start+(common_setup if index == 0 and trial == 0 else 0)
                    if trial == 0:
                        times[mode] += oracle_seconds
                    guards[mode] = guards[mode] and after == before
                    peaks[mode] = torch.cuda.max_memory_reserved()
                    if peaks[mode] > 4*1024**3:
                        raise MemoryError("The complete native loader exceeded its CUDA budget")
                row = {"source_index": index, "trial": trial, "source": source_ids[path], "complete_stage_seconds": times,
                    "all_full_states_devices_layouts_aliases_and_source_bytes_exact": all(guards.values()),
                    "CUDA_reserved_peaks": peaks, "native_capture_tensors": payload_digest}
                pairs.append(row)
                atomic_json(args.output/"STATE.json", {"status": "running", "pairs": pairs, "counted": False})
                print({k: row[k] for k in ("source_index", "trial", "complete_stage_seconds", "all_full_states_devices_layouts_aliases_and_source_bytes_exact")}, flush=True)
                if not all(guards.values()):
                    raise ValueError("A complete native private-map state or ownership guard failed")
            del oracle
            gc.collect()
            torch.cuda.empty_cache()
        by_source = [{"source_index": index, "median_seconds": {mode: statistics.median(p["complete_stage_seconds"][mode]
            for p in pairs if p["source_index"] == index) for mode in ("native", "mapped")}} for index in range(3)]
        sums = {mode: sum(p["median_seconds"][mode] for p in by_source) for mode in ("native", "mapped")}
        reduction = 1-sums["mapped"]/sums["native"]
        checks = {"all_three_real_sources_all_three_pairs_native_values_devices_aliases_immutable": len(pairs) == 9
                and all(p["all_full_states_devices_layouts_aliases_and_source_bytes_exact"] for p in pairs),
            "complete_stage_sum_medians_reduces10percent": reduction >= .1,
            "sources_inputs_unchanged": identity == [data_identity(p) for p in paths]}
        result = {"status": "native_private_map_stage_passed_recovery_pending" if all(checks.values()) else "failed_verification",
            "checks": checks, "pairs": pairs, "by_source": by_source, "sum_source_median_seconds": sums,
            "reduction": reduction, "common_CUDA_setup_in_both_first_costs": common_setup, "sources_inputs": identity,
            "actual_native_restore_or_training_executed": False, "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        print({k: result[k] for k in ("status", "checks", "sum_source_median_seconds", "reduction")}, flush=True)
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"error": repr(error), "pairs": pairs, "counted": False})
        raise


if __name__ == "__main__":
    main()
