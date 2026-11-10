"""Batch live Point3D predicates without replacing native update or mutation."""

from contextlib import contextmanager
from pathlib import Path
import sys
import time

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
import export_admitted_map as native

ORIGINAL_FILTER = native.filter_points
MAXIMUM_TEMPORARY_BYTES = 256 * 1024**2


def admission(model, policy, lo, hi):
    """Choose the original route before mutation for unsupported interfaces."""
    if pycolmap.__version__ != "4.1.1" or type(model) is not pycolmap.Reconstruction:
        return "unsupported_native_type_or_version"
    if type(policy) is not dict:
        return "unsupported_policy_type"
    maximum = policy.get("maximum_point_reprojection_error_px")
    minimum = policy.get("minimum_point_track_length")
    if (type(maximum) not in (int, float)
            or (type(maximum) is int and abs(maximum) > 2**53)
            or (type(maximum) is float and not np.isfinite(maximum))
            or type(minimum) is not int or not 0 <= minimum <= 2**31):
        return "unsupported_policy_scalar"
    if any(type(bound) is not np.ndarray or bound.dtype != np.dtype("float64")
           or bound.shape != (3,) or not np.isfinite(bound).all()
           for bound in (lo, hi)):
        return "unsupported_extent_interface"
    if np.any(lo > hi):
        return "unsupported_extent_order"
    # Reserve conservative per-point space for wrappers, IDs, and owned arrays.
    if model.num_points3D() * 1024 > MAXIMUM_TEMPORARY_BYTES:
        return "temporary_memory_admission_exceeded"
    return None


def filter_points_columns(model, policy, lo, hi, original=ORIGINAL_FILTER, detail=None):
    """Rebuild owned columns for this invocation and retain sequential deletion."""
    detail = {} if detail is None else detail
    abstain = admission(model, policy, lo, hi)
    if abstain is not None:
        detail.update(route="original", abstain=abstain)
        return original(model, policy, lo, hi)
    detail.update(route="columns", abstain=None)
    model.update_point_3d_errors()
    point_ids = list(model.points3D)
    points = [model.points3D[point_id] for point_id in point_ids]
    count = len(points)
    xyz = np.array([point.xyz for point in points], dtype=np.float64).reshape(count, 3)
    errors = np.fromiter((point.error for point in points), dtype=np.float64, count=count)
    lengths = np.fromiter((point.track.length() for point in points), dtype=np.int64, count=count)
    invalid_extent = (~np.isfinite(xyz).all(axis=1)
                      | np.any(xyz < lo, axis=1) | np.any(xyz > hi, axis=1))
    invalid_error = (~np.isfinite(errors)
                     | (errors > policy["maximum_point_reprojection_error_px"]))
    invalid_track = lengths < policy["minimum_point_track_length"]
    detail.update(input_point_count=count, owned_column_bytes=xyz.nbytes + errors.nbytes + lengths.nbytes,
                  owned_xyz=bool(xyz.flags.owndata or xyz.base is not None and xyz.base.flags.owndata),
                  owned_errors=bool(errors.flags.owndata), owned_lengths=bool(lengths.flags.owndata))
    removed = []
    for index, point_id in enumerate(point_ids):
        reasons = []
        if invalid_extent[index]:
            reasons.append("invalid_or_outside_metric_extent")
        if invalid_error[index]:
            reasons.append("reprojection_error")
        if invalid_track[index]:
            reasons.append("track_length")
        if reasons:
            point = model.points3D[point_id]
            removed.append({"point3D_id": int(point_id), "reasons": reasons,
                "observations": [[int(e.image_id), int(e.point2D_idx)] for e in point.track.elements]})
            model.delete_point3D(point_id)
    detail.update(removed_count=len(removed), remaining_point_count=model.num_points3D())
    return removed


@contextmanager
def point_filter_execution(module=native, enabled=True):
    started = time.perf_counter()
    original = module.filter_points
    if original is not ORIGINAL_FILTER or Path(module.__file__).resolve() != Path(native.__file__).resolve():
        raise ValueError("The original point-filter binding is not the frozen native entry")
    trace = {"calls": [], "enabled": bool(enabled)}

    def execute(model, policy, lo, hi):
        begin = time.perf_counter()
        detail = {"route": "original"}
        try:
            result = (filter_points_columns(model, policy, lo, hi, original, detail)
                      if enabled else original(model, policy, lo, hi))
            detail.update(status="original_filter_completed", removed_count=len(result))
            return result
        except BaseException as error:
            detail.update(status="original_filter_error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            detail["complete_call_seconds"] = time.perf_counter() - begin
            trace["calls"].append(detail)

    module.filter_points = execute
    trace["setup_seconds"] = time.perf_counter() - started
    try:
        yield trace
    finally:
        begin = time.perf_counter()
        module.filter_points = original
        trace["restore_seconds"] = time.perf_counter() - begin
        trace["original_binding_restored"] = module.filter_points is original
