"""Defer CUDA Boolean materialization within one passive numeric comparison."""

from contextlib import contextmanager


def native_item_path(torch, left, right):
    if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.device.type == "cuda" and right.device == left.device
            and left.shape == right.shape and left.dtype == right.dtype
            and left.layout == right.layout == torch.strided and left.numel()
            and not any(name is not None for name in left.names + right.names)):
        return False
    alias = (left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset()
        and left.is_contiguous() == right.is_contiguous()
        and left.stride() == right.stride()
        and left.is_conj() == right.is_conj() and left.is_neg() == right.is_neg())
    return not alias


def passive(torch, np, value, devices, depth=0):
    if depth > 48:
        return False
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        if (value.layout != torch.strided or value.is_quantized or value.is_conj() or value.is_neg()
                or any(name is not None for name in value.names)
                or value.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16,
                    torch.int32, torch.int64, torch.float16, torch.bfloat16,
                    torch.float32, torch.float64, torch.complex64, torch.complex128)):
            return False
        if value.device.type == "cuda":
            if value.numel() > 64 * 1024**2:
                return False
            devices.add(value.device)
        elif value.device.type != "cpu":
            return False
        return True
    if type(value) is np.ndarray:
        return value.dtype.kind in "buifc"
    if isinstance(value, np.generic) and type(value).__module__ == "numpy":
        return value.dtype.kind in "buifc"
    if type(value) is dict:
        return all(type(key) in (str, bytes, int, bool) and passive(torch, np, child, devices, depth + 1)
            for key, child in value.items())
    if type(value) in (list, tuple):
        return all(passive(torch, np, child, devices, depth + 1) for child in value)
    return type(value) in (type(None), str, bytes, bool, int, float)


@contextmanager
def deferred_checkpoint_equal(adapter, enabled):
    import numpy as np
    import torch

    if type(enabled) is not bool:
        raise TypeError("The equality mechanism requires an explicit boolean flag")
    original_function = adapter.exact_equal
    original_tensor_equal = torch.equal
    rows = []
    trace = {"enabled": enabled, "calls": 0, "supported_tree_calls": 0,
        "original_route_calls": 0, "CUDA_boolean_materializations": 0,
        "CUDA_comparison_leaves": 0, "maximum_pending_scalar_flags": 0,
        "no_payload_or_optimizer_mutation": True, "rows": rows,
        "materialization_counts_use_fixed_native_item_path_not_total_GPU_operations": True}

    def native_equal(left, right):
        result = original_tensor_equal(left, right)
        cuda = isinstance(left, torch.Tensor) and left.device.type == "cuda"
        population = int(native_item_path(torch, left, right))
        trace["CUDA_boolean_materializations"] += population
        trace["CUDA_comparison_leaves"] += int(cuda)
        rows.append({"route": "original_tensor_equal", "CUDA_left": cuda,
            "shape": list(left.shape), "dtype": str(left.dtype),
            "device": str(left.device), "item_population": population, "result": result})
        return result

    def compare(left, right):
        trace["calls"] += 1
        devices = set()
        if not enabled or not passive(torch, np, left, devices) or not passive(torch, np, right, devices) or len(devices) > 1:
            trace["original_route_calls"] += 1
            return original_function(left, right)
        trace["supported_tree_calls"] += 1
        pending = []

        def visit(first, second):
            if isinstance(first, torch.Tensor):
                if not isinstance(second, torch.Tensor) or first.shape != second.shape or first.dtype != second.dtype:
                    return False
                if first.device != second.device or first.device.type != "cuda" or not native_item_path(torch, first, second):
                    return native_equal(first, second)
                pending.append(torch.eq(first, second).all())
                trace["CUDA_comparison_leaves"] += 1
                trace["maximum_pending_scalar_flags"] = max(trace["maximum_pending_scalar_flags"], len(pending))
                rows.append({"route": "deferred_CUDA_eq_all", "CUDA_left": True,
                    "shape": list(first.shape), "dtype": str(first.dtype), "device": str(first.device),
                    "item_population": 0, "result_not_individually_materialized": True})
                return True
            if isinstance(first, np.ndarray):
                return isinstance(second, np.ndarray) and np.array_equal(first, second)
            if isinstance(first, dict):
                return isinstance(second, dict) and first.keys() == second.keys() and all(visit(first[key], second[key]) for key in first)
            if isinstance(first, (tuple, list)):
                return type(first) is type(second) and len(first) == len(second) and all(visit(a, b) for a, b in zip(first, second))
            return first == second

        metadata_and_CPU = visit(left, right)
        if not metadata_and_CPU:
            return False
        if pending:
            result = torch.stack(pending).all().item()
            trace["CUDA_boolean_materializations"] += 1
            rows.append({"route": "single_deferred_CUDA_bool_item", "pending_flags": len(pending),
                "item_population": 1, "result": result})
            return result
        return metadata_and_CPU

    torch.equal = native_equal
    adapter.exact_equal = compare
    try:
        yield trace
    except BaseException:
        if trace["CUDA_comparison_leaves"]:
            torch.cuda.synchronize()
            trace["exception_owned_current_stream_drained"] = True
        raise
    finally:
        adapter.exact_equal = original_function
        torch.equal = original_tensor_equal
        trace["original_comparator_and_tensor_equal_bindings_restored"] = adapter.exact_equal is original_function and torch.equal is original_tensor_equal

