"""Create only the native temporary SH zero features on the current GPU."""

from contextlib import contextmanager
import difflib
import inspect
import json
from pathlib import Path
import textwrap
import time


@contextmanager
def native_feature_zero(output, enabled):
    import torch
    from scene.gaussian_model import GaussianModel

    output = Path(output)
    original = GaussianModel.create_from_pcd
    native_cuda = torch.Tensor.cuda
    source = textwrap.dedent(inspect.getsource(original))
    anchor = "features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()"
    if source.count(anchor) != 1:
        raise ValueError("The pinned native zero-feature expression changed")
    changed = source.replace(anchor, "features = _owned_native_feature_zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))")
    trace = {"enabled": enabled, "calls": [], "requested_upload_bytes": 0,
             "actual_profiler_HtoD_bytes": 0, "no_point_upload_reuse": True,
             "native_functions_restored": False, "new_optimization_count": 0}

    def zeros(shape):
        return torch.zeros(shape, dtype=torch.float32, device="cuda")

    namespace = dict(original.__globals__)
    namespace["_owned_native_feature_zeros"] = zeros
    exec(compile(changed, str(output / "EXECUTED_CREATE_FROM_PCD.py"), "exec"), namespace)
    candidate = namespace["create_from_pcd"]
    (output / "EXECUTED_CREATE_FROM_PCD.py").write_text(changed, encoding="utf-8")
    (output / "CREATE_FROM_PCD.diff").write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="pinned_native", tofile="private_direct_zero")), encoding="utf-8")
    active = []

    def cuda(tensor, *args, **kwargs):
        result = native_cuda(tensor, *args, **kwargs)
        if active and tensor.device.type == "cpu" and result.device.type == "cuda":
            row = {"source_dtype": str(tensor.dtype), "source_device": str(tensor.device),
                   "shape": list(tensor.shape), "stride": list(tensor.stride()),
                   "result_device": str(result.device),
                   "payload_bytes": tensor.numel() * tensor.element_size()}
            active[-1]["uploads"].append(row)
        return result

    def create(model, pcd, cam_infos, spatial_lr_scale):
        if active:
            raise RuntimeError("Native zero-feature ownership cannot be nested")
        supported = (torch.get_default_dtype() == torch.float32
                     and torch.get_default_device().type == "cpu"
                     and type(model.max_sh_degree) is int and 0 <= model.max_sh_degree <= 3)
        selected = candidate if enabled and supported else original
        row = {"route": "direct_zero" if selected is candidate else "native",
               "supported_default_and_degree": supported, "uploads": [], "status": "running"}
        trace["calls"].append(row)
        number = len(trace["calls"])
        trace_file = output / (f"CUDA_TRANSFER_{number:03d}.json")
        begin = time.perf_counter()
        active.append(row)
        profiler = None
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                torch.cuda.synchronize()
                selected(model, pcd, cam_infos, spatial_lr_scale)
                torch.cuda.synchronize()
            row["status"] = "complete"
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            active.pop()
            row["complete_profiled_initializer_seconds"] = time.perf_counter() - begin
            if profiler is not None:
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
            row["requested_upload_bytes"] = sum(item["payload_bytes"] for item in row["uploads"])
            row["profiler_matches_all_native_uploads"] = row["actual_profiler_HtoD_bytes"] == row["requested_upload_bytes"]
            trace["requested_upload_bytes"] += row["requested_upload_bytes"]
            trace["actual_profiler_HtoD_bytes"] += row["actual_profiler_HtoD_bytes"]

    try:
        GaussianModel.create_from_pcd = create
        torch.Tensor.cuda = cuda
        yield trace
    finally:
        GaussianModel.create_from_pcd = original
        torch.Tensor.cuda = native_cuda
        trace["native_functions_restored"] = (GaussianModel.create_from_pcd is original
                                               and torch.Tensor.cuda is native_cuda and not active)
        (output / "ZERO_TRANSFER_TRACE.json").write_text(json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")
