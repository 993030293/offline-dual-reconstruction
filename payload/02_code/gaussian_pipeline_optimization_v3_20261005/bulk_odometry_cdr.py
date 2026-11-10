"""Decode the same contiguous native CDR float64 fields with two bounded structs."""

from contextlib import contextmanager
import inspect
import struct

import run_effective_initial

LAYOUTS = {7: struct.Struct("<7d"), 36: struct.Struct("<36d")}


def decode_array(cdr, count):
    if count not in LAYOUTS:
        raise ValueError("Only the registered native pose and covariance arrays are supported")
    cdr.align(8)
    values = list(LAYOUTS[count].unpack_from(cdr.data, cdr.offset))
    cdr.offset += LAYOUTS[count].size
    return values


@contextmanager
def bulk_odometry_execution(enabled=True):
    original = run_effective_initial.combined_raw_execution
    source = inspect.getsource(original)
    anchor = "    namespace.update(_effective_pairs=pairs, _effective_exports=exports)"
    after = '''    trace["bulk_CDR"] = {"pose_calls": 0, "covariance_calls": 0}
    def bulk(cdr, count):
        key = "pose_calls" if count == 7 else "covariance_calls"
        trace["bulk_CDR"][key] += 1
        return _bulk_cdr_decoder(cdr, count)
    before_pose = '        poses.append([cdr.f64() for _ in range(7)])'
    before_covariance = '        covariance.append([cdr.f64() for _ in range(36)])'
    if source.count(before_pose) != 1 or source.count(before_covariance) != 1:
        raise ValueError("The native CDR pose/covariance field call sites changed")
    source = source.replace(before_pose, '        poses.append(_effective_CDR(cdr, 7))')
    source = source.replace(before_covariance, '        covariance.append(_effective_CDR(cdr, 36))')
    namespace.update(_effective_pairs=pairs, _effective_exports=exports, _effective_CDR=bulk)'''
    if source.count(anchor) != 1:
        raise ValueError("The single-source accepted raw adapter composition changed")
    namespace = dict(run_effective_initial.__dict__)
    namespace["_bulk_cdr_decoder"] = decode_array
    if enabled:
        exec(compile(source.replace(anchor, after), "<accepted_raw_combination_with_exact_bulk_CDR>", "exec"), namespace)
        run_effective_initial.combined_raw_execution = namespace["combined_raw_execution"]
    try:
        yield
    finally:
        run_effective_initial.combined_raw_execution = original
