"""Reuse the native read-only distance operand after its first GPU upload."""

from contextlib import contextmanager
import difflib
import inspect
import json
from pathlib import Path
import textwrap
import time
import sys

from native_transfer_meter import NativeTransferMeter


@contextmanager
def native_coordinate_reuse(output, enabled):
    import numpy as np
    import torch
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud

    output = Path(output)
    original = GaussianModel.create_from_pcd
    native_cuda = torch.Tensor.cuda
    source = textwrap.dedent(inspect.getsource(original))
    zero = "features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()"
    operand = "distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda())"
    if source.count(zero) != 1 or source.count(operand) != 1:
        raise ValueError("The pinned native initialization operands changed")
    incumbent = source.replace(zero, "features = _owned_zero((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))")
    changed = incumbent.replace(operand, "distCUDA2(fused_point_cloud)")
    namespace = dict(original.__globals__)
    namespace["_owned_zero"] = lambda shape: torch.zeros(shape, dtype=torch.float32, device="cuda")
    functions = {}
    for name, body in (("direct_zero", incumbent), ("coordinate_reuse", changed)):
        exec(compile(body, str(output / (name + "_CREATE_FROM_PCD.py")), "exec"), namespace)
        functions[name] = namespace["create_from_pcd"]
        (output / (name + "_CREATE_FROM_PCD.py")).write_text(body, encoding="utf-8")
    (output / "COORDINATE_REUSE.diff").write_text("".join(difflib.unified_diff(
        incumbent.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="common_R157_incumbent", tofile="private_operand_reuse")), encoding="utf-8")
    meter = NativeTransferMeter()
    active = []
    trace = {"enabled": enabled, "common_R157_zero": True, "calls": [],
             "actual_native_HtoD_bytes": 0, "native_functions_restored": False,
             "new_optimization_count": 0}

    def cuda(tensor, *args, **kwargs):
        result = native_cuda(tensor, *args, **kwargs)
        if active and tensor.device.type == "cpu" and result.device.type == "cuda":
            active[-1]["uploads"].append({"source_dtype": str(tensor.dtype),
                "source_device": str(tensor.device), "shape": list(tensor.shape),
                "stride": list(tensor.stride()), "result_device": str(result.device),
                "payload_bytes": tensor.numel() * tensor.element_size()})
        return result

    def aliases(model):
        groups, result = {}, {}
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation",
                     "_exposure", "max_radii2D", "xyz_gradient_accum", "denom"):
            value = getattr(model, name)
            key = str(value.device), value.untyped_storage().data_ptr()
            result[name] = groups.setdefault(key, len(groups))
        return result

    def create(model, pcd, cam_infos, spatial_lr_scale):
        if active:
            raise RuntimeError("Private coordinate operand ownership cannot be nested")
        zero_supported = (torch.get_default_dtype() == torch.float32
                          and torch.get_default_device().type == "cpu"
                          and type(model.max_sh_degree) is int and 0 <= model.max_sh_degree <= 3)
        points = pcd.points if type(pcd) is BasicPointCloud else None
        supported = (zero_supported and type(points) is np.ndarray and points.dtype == np.float32
                     and points.ndim == 2 and points.shape[1] == 3 and points.shape[0] >= 4
                     and (points.flags.c_contiguous or points.flags.f_contiguous)
                     and points.flags.writeable and bool(np.isfinite(points).all()))
        route = "coordinate_reuse" if enabled and supported else "direct_zero" if zero_supported else "native"
        selected = functions.get(route, original)
        row = {"route": route, "supported_operand": supported, "uploads": [], "status": "running"}
        trace["calls"].append(row)
        active.append(row)
        begin = time.perf_counter()
        meter.begin()
        try:
            torch.cuda.synchronize()
            selected(model, pcd, cam_infos, spatial_lr_scale)
            torch.cuda.synchronize()
            row["parameter_storage_alias_groups"] = aliases(model)
            row["status"] = "complete"
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            events = meter.end()
            row["actual_native_events"] = events
            row["native_backend"] = meter.backend
            row["actual_native_HtoD_bytes"] = sum(event["bytes"] for event in events
                                                   if event["kind"] == 1 and event["result"] == 0)
            row["requested_upload_bytes"] = sum(item["payload_bytes"] for item in row["uploads"])
            row["native_API_matches_all_tensor_uploads"] = row["requested_upload_bytes"] == row["actual_native_HtoD_bytes"]
            row["complete_metered_initializer_seconds"] = time.perf_counter() - begin
            trace["actual_native_HtoD_bytes"] += row["actual_native_HtoD_bytes"]
            active.pop()
            if any(event["result"] for event in events):
                error = sys.exception()
                if error is None:
                    raise RuntimeError("A native CUDA copy failed inside the private operand scope")
                error.add_note("The native transfer meter also recorded a failed CUDA copy")
    try:
        GaussianModel.create_from_pcd = create
        torch.Tensor.cuda = cuda
        yield trace
    finally:
        GaussianModel.create_from_pcd = original
        torch.Tensor.cuda = native_cuda
        trace["native_functions_restored"] = (GaussianModel.create_from_pcd is original
                                               and torch.Tensor.cuda is native_cuda and not active)
        (output / "COORDINATE_TRANSFER_TRACE.json").write_text(
            json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")
