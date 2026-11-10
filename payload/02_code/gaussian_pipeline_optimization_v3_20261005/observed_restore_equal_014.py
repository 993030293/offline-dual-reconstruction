"""Observe original equality calls and their CPU profiler scalar events."""

from contextlib import contextmanager
import time


@contextmanager
def observe_restore_equality(torch):
    original = torch.equal
    marker = "R014_original_CUDA_equal"
    rows = []
    trace = {"rows": rows, "algorithm_not_modified": True,
        "CPU_operator_profile_not_CUDA_kernel_or_transfer_timing": True}
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])

    def observed(left, right):
        started = time.perf_counter()
        eligible = isinstance(left, torch.Tensor) and left.is_cuda
        metadata = {"left_shape": list(left.shape), "left_dtype": str(left.dtype),
            "left_device": str(left.device), "left_numel": left.numel(),
            "right_shape": list(right.shape), "right_dtype": str(right.dtype),
            "right_device": str(right.device), "CUDA_left": eligible,
            "same_storage": left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()}
        if eligible:
            with torch.autograd.profiler.record_function(marker):
                result = original(left, right)
        else:
            result = original(left, right)
        rows.append({**metadata, "result": result, "observed_API_seconds_including_instrumentation": time.perf_counter() - started})
        return result

    torch.equal = observed
    try:
        with profiler:
            yield trace
    finally:
        torch.equal = original
        trace["original_torch_equal_binding_restored"] = torch.equal is original
        counts = {}
        marked = []
        for event in profiler.events():
            parent = event.cpu_parent
            under = event.name == marker
            while parent is not None and not under:
                under = parent.name == marker
                parent = parent.cpu_parent
            if under:
                counts[event.name] = counts.get(event.name, 0) + 1
                if event.name in ("aten::item", "aten::_local_scalar_dense", "aten::equal"):
                    marked.append({"name": event.name, "CPU_total_microseconds": event.cpu_time_total})
        trace["native_CUDA_equal_CPU_operator_counts"] = counts
        trace["native_CUDA_equal_scalar_operator_events"] = marked
        trace["native_CUDA_equal_host_item_events"] = counts.get("aten::item", 0)
        trace["native_CUDA_equal_local_scalar_dense_events"] = counts.get("aten::_local_scalar_dense", 0)
        trace["native_CUDA_equal_calls"] = sum(row["CUDA_left"] for row in rows)
        trace["profiler_context_closed"] = True

