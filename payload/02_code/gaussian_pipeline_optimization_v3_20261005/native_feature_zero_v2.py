"""Retain V1's exact mechanism while reading successful native copy payloads."""

import difflib
import inspect
from pathlib import Path
import sys

from native_feature_zero import native_feature_zero as preserved_v1
from native_transfer_meter import NativeTransferMeter


def native_feature_zero(output, enabled):
    native = inspect.unwrap(preserved_v1)
    source = inspect.getsource(native)
    before = '''        profiler = None
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                torch.cuda.synchronize()
                selected(model, pcd, cam_infos, spatial_lr_scale)
                torch.cuda.synchronize()
'''
    after = '''        meter.begin()
        try:
            torch.cuda.synchronize()
            selected(model, pcd, cam_infos, spatial_lr_scale)
            torch.cuda.synchronize()
'''
    block = '''            if profiler is not None:
                profiler.export_chrome_trace(str(trace_file))
                events = json.loads(trace_file.read_text(encoding="utf-8"))["traceEvents"]
                transfers = [event for event in events if event.get("cat") == "gpu_memcpy"
                             and "HtoD" in event.get("name", "")]
                if any(type(event.get("args", {}).get("bytes")) is not int
                       or event["args"]["bytes"] < 0 for event in transfers):
                    raise ValueError("Actual CUDA memcpy events omitted valid transfer bytes")
                row["actual_memcpy_events"] = [{"name": event["name"], "bytes": event["args"]["bytes"]}
                                               for event in transfers]
                row["actual_profiler_HtoD_bytes"] = sum(event["args"]["bytes"] for event in transfers)
            else:
                row["actual_profiler_HtoD_bytes"] = 0
'''
    native_block = '''            events = meter.end()
            row["actual_native_events"] = events
            row["native_backend"] = meter.backend
            row["actual_profiler_HtoD_bytes"] = sum(event["bytes"] for event in events
                                                    if event["kind"] == 1 and event["result"] == 0)
            if any(event["result"] for event in events):
                active_error = _owned_current_exception()
                if active_error is None:
                    raise RuntimeError("An actual native CUDA copy failed within the scope")
                active_error.add_note("The native meter also recorded a failed CUDA copy")
'''
    replacements = (("    native_cuda = torch.Tensor.cuda", "    native_cuda = torch.Tensor.cuda\n    meter = NativeTransferMeter()"),
                    (before, after), (block, native_block),
                    ('            row["status"] = "complete"', '            row["parameter_storage_alias_groups"] = _owned_parameter_aliases(model)\n            row["status"] = "complete"'))
    changed = source
    for anchor, replacement in replacements:
        if changed.count(anchor) != 1:
            raise ValueError("The preserved V1 native meter boundary changed")
        changed = changed.replace(anchor, replacement)
    changed = changed.replace("actual_profiler_HtoD_bytes", "actual_native_HtoD_bytes")
    changed = changed.replace("profiler_matches_all_native_uploads", "native_API_matches_all_tensor_uploads")
    changed = changed.replace("complete_profiled_initializer_seconds", "complete_metered_initializer_seconds")
    namespace = dict(native.__globals__)
    namespace["NativeTransferMeter"] = NativeTransferMeter
    namespace["_owned_current_exception"] = sys.exception
    def aliases(model):
        groups, result = {}, {}
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation",
                     "_exposure", "max_radii2D", "xyz_gradient_accum", "denom"):
            value = getattr(model, name)
            key = str(value.device), value.untyped_storage().data_ptr()
            if key not in groups:
                groups[key] = len(groups)
            result[name] = groups[key]
        return result
    namespace["_owned_parameter_aliases"] = aliases
    exec(compile(changed, str(Path(output) / "EXECUTED_METER_CONTEXT.py"), "exec"), namespace)
    (Path(output) / "EXECUTED_METER_CONTEXT.py").write_text(changed, encoding="utf-8")
    (Path(output) / "METER_CONTEXT.diff").write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="preserved_CUPTI_V1", tofile="native_runtime_V2")), encoding="utf-8")
    return namespace["native_feature_zero"](output, enabled)
