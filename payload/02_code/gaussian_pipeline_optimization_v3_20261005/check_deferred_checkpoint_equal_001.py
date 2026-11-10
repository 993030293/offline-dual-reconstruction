"""Check original equality semantics and actual scalar events on real CUDA."""

import json
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from deferred_checkpoint_equal_001 import deferred_checkpoint_equal


def attempt(function):
    try:
        value = function()
        return {"kind": "returned", "type": type(value).__name__, "value": bool(value)}
    except Exception as error:
        return {"kind": "raised", "type": type(error).__name__, "message": str(error)}


class RaisingValue:
    def __eq__(self, other):
        raise RuntimeError("original_custom_equality_failure")


def main():
    import numpy as np
    import torch
    import checkpoint_adapter as adapter

    output = Path(sys.argv[1]).resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014/R168")
    output.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != "2.7.1+cu128":
        raise ValueError("The current Torch interface changed")
    original_equal, original_function = torch.equal, adapter.exact_equal
    public_path = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ocd001/public_inputs001/RESULT.json"
    originals = [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("deferred_checkpoint_equal_001.py"),
        ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/checkpoint_adapter.py", public_path)]
    atomic_json(output / "PROTOCOL.json", {"sources_inputs": originals, "parent_count": 36, "new_optimization_count": 0,
        "current_real_CUDA_CPU_operator_evidence_not_full_restore_or_GT": True})
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()})
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * 1024**3:
        raise MemoryError("The original GPU reserve is unavailable")
    torch.cuda.reset_peak_memory_stats()
    fixtures = []
    for dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128):
        value = torch.arange(48, device="cuda", dtype=torch.int32).reshape(6, 8).to(dtype)
        fixtures.append(("equal_" + str(dtype), value, value.clone()))
        first = value.clone()
        second = value.clone()
        second[0, 0] = 1
        fixtures.append(("different_" + str(dtype), first, second))
    nan = torch.tensor([float("nan"), 1.0], device="cuda")
    shared = torch.arange(8, device="cuda", dtype=torch.float32)
    noncontiguous = torch.arange(48, device="cuda", dtype=torch.float32).reshape(6, 8).t()
    complex_value = torch.tensor([1 + 2j, 3 - 4j], device="cuda")
    fixtures.extend([
        ("NaN_nonalias_original_false", nan, nan.clone()),
        ("NaN_alias_original_fast_path", nan, nan),
        ("shared_storage_different_offsets", shared[:4], shared[4:]),
        ("noncontiguous", noncontiguous, noncontiguous.clone()),
        ("empty", torch.empty(0, 3, device="cuda"), torch.empty(0, 3, device="cuda")),
        ("signed_zero_and_infinity", torch.tensor([-0., float("inf")], device="cuda"), torch.tensor([0., float("inf")], device="cuda")),
        ("shape_mismatch", shared, shared.reshape(2, 4)),
        ("dtype_mismatch", shared, shared.to(torch.float64)),
        ("cross_device_original_error", shared, shared.cpu()),
        ("conjugate_original_route", complex_value.conj(), complex_value.conj().clone()),
        ("named_original_route", shared.refine_names("sample"), shared.clone().refine_names("sample")),
        ("mixed_numeric_tree", {"model": (shared, shared.clone()), "rng": (np.arange(32, dtype=np.uint32), torch.arange(16)), "scale": np.float64(1.25)},
            {"model": (shared.clone(), shared.clone()), "rng": (np.arange(32, dtype=np.uint32), torch.arange(16)), "scale": np.float64(1.25)}),
        ("dict_keys", {"a": shared}, {"b": shared.clone()}),
        ("tuple_list_types", (shared,), [shared.clone()]),
        ("unsupported_exception", (shared, RaisingValue()), (shared.clone(), RaisingValue())),
        ("original_short_circuit_before_custom_value", (shared, RaisingValue()), (shared + 1, RaisingValue())),
        ("numpy_scalar_return_type", np.float64(1), np.float64(1)),
    ])
    rows = []
    for name, left, right in fixtures:
        baseline = attempt(lambda: original_function(left, right))
        with deferred_checkpoint_equal(adapter, True) as trace:
            candidate = attempt(lambda: adapter.exact_equal(left, right))
        rows.append({"name": name, "baseline": baseline, "candidate": candidate,
            "same": baseline == candidate, "bindings_restored": trace["original_comparator_and_tensor_equal_bindings_restored"]})
    actual = []
    tensor_tree = [torch.arange(256, device="cuda", dtype=torch.float32) + k for k in range(22)]
    tensor_copy = [value.clone() for value in tensor_tree]
    for enabled in (False, True):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            with deferred_checkpoint_equal(adapter, enabled) as trace:
                value = adapter.exact_equal(tensor_tree, tensor_copy)
        item_events = sum(event.name == "aten::item" for event in profiler.events())
        actual.append({"enabled": enabled, "value": value, "actual_CPU_profiler_item_events": item_events,
            "recorded_materializations": trace["CUDA_boolean_materializations"], "CUDA_leaves": trace["CUDA_comparison_leaves"],
            "bindings_restored": trace["original_comparator_and_tensor_equal_bindings_restored"]})
    public = json.loads(public_path.read_text())
    if len(public["poses"]) != 360:
        raise ValueError("The entire registered public development pose denominator changed")
    tree = {"names": [row["name"] for row in public["poses"]],
        "position": torch.tensor([row["position"] for row in public["poses"]], dtype=torch.float64, device="cuda"),
        "rotation": torch.tensor([row["rotation_c2w"] for row in public["poses"]], dtype=torch.float64, device="cuda")}
    other = {key: value.clone() if isinstance(value, torch.Tensor) else list(value) for key, value in tree.items()}
    public_results = []
    for label in ("all360_exact", "one_last_pose_changed"):
        if label != "all360_exact":
            other["position"][-1, 0] += 1
        before = attempt(lambda: original_function(tree, other))
        with deferred_checkpoint_equal(adapter, True) as trace:
            after = attempt(lambda: adapter.exact_equal(tree, other))
        public_results.append({"label": label, "baseline": before, "candidate": after,
            "same": before == after, "bindings_restored": trace["original_comparator_and_tensor_equal_bindings_restored"]})
    torch.cuda.synchronize()
    checks = {
        "all41_real_CUDA_numeric_type_metadata_layout_alias_and_error_fixtures_same": len(rows) == 41 and all(row["same"] and row["bindings_restored"] for row in rows),
        "NaN_alias_fast_path_preserved_true_and_nonalias_false": next(row for row in rows if row["name"] == "NaN_alias_original_fast_path")["candidate"]["value"] and not next(row for row in rows if row["name"] == "NaN_nonalias_original_false")["candidate"]["value"],
        "actual_native22_and_candidate1_profiler_item_events_match_recorded_population": [row["actual_CPU_profiler_item_events"] for row in actual] == [22, 1] and all(row["recorded_materializations"] == row["actual_CPU_profiler_item_events"] and row["CUDA_leaves"] == 22 and row["value"] for row in actual),
        "all360_public_numeric_components_exact_and_last_pose_difference_detected": all(row["same"] and row["bindings_restored"] for row in public_results) and public_results[0]["candidate"]["value"] and not public_results[1]["candidate"]["value"],
        "all_original_functions_and_inputs_preserved": torch.equal is original_equal and adapter.exact_equal is original_function and originals == [data_identity(row["path"]) for row in originals],
        "native_GPU12GiB_and_free2GiB_reserve": torch.cuda.max_memory_reserved() <= 12 * 1024**3 and torch.cuda.mem_get_info()[0] >= 2 * 1024**3,
    }
    atomic_json(output / "RESULT.json", {"checks": checks, "fixtures": rows, "actual_operator_profiles": actual,
        "public_all360_components": public_results, "sources_inputs": originals, "new_optimization_count": 0,
        "full_restore_formal_error_cancel_recovery_not_this_component": True,
        "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved()})
    print(json.dumps({"checks": checks, "fixtures": len(rows), "actual_item_events": [row["actual_CPU_profiler_item_events"] for row in actual], "new_count": 0}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The original equality semantics or actual work failed verification")


if __name__ == "__main__":
    main()
