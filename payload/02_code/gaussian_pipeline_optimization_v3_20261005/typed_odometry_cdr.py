"""Decode complete native trajectories into bounded contiguous typed buffers."""

from contextlib import contextmanager
import inspect
import struct

import numpy as np

import run_effective_initial

MAXIMUM_RECORDS = 500000
MAXIMUM_BUFFER_BYTES = 256*1024**2


def admitted(count):
    return type(count) is int and 0 <= count <= MAXIMUM_RECORDS and count*352 <= MAXIMUM_BUFFER_BYTES


def buffers(count):
    if not admitted(count):
        raise ValueError("Select the original trajectory lists before exceeding the typed buffer bound")
    return np.empty(count, dtype=np.int64), np.empty((count, 7), dtype=np.float64), np.empty((count, 36), dtype=np.float64), set()


def decode_view(cdr, count):
    if count not in (7, 36):
        raise ValueError("Only the complete native pose and covariance fields are registered")
    cdr.align(8)
    if cdr.offset+count*8 > len(cdr.data):
        raise struct.error("The complete native CDR float64 array is truncated")
    result = np.frombuffer(cdr.data, dtype="<f8", count=count, offset=cdr.offset)
    cdr.offset += count*8
    return result


def complete(stamps, poses, covariance, seen):
    if not (seen == len(stamps) and poses.shape == (seen, 7) and covariance.shape == (seen, 36)):
        raise ValueError("The indexed raw trajectory did not fill every owned typed record")


@contextmanager
def typed_odometry_execution(enabled=True):
    original = run_effective_initial.combined_raw_execution
    factory = inspect.getsource(original)
    anchor = "    namespace.update(_effective_pairs=pairs, _effective_exports=exports)"
    after = '''    trace["typed_CDR"] = {"admission_calls": 0, "native_route_calls": 0, "typed_route_calls": 0,
        "pose_calls": 0, "covariance_calls": 0, "complete_records": 0, "owned_typed_buffer_bytes": 0,
        "all36_covariance_values_retained": True, "maximum_records": _typed_maximum_records}
    def admit(count):
        trace["typed_CDR"]["admission_calls"] += 1
        allowed = _typed_admission(count)
        trace["typed_CDR"]["typed_route_calls" if allowed else "native_route_calls"] += 1
        return allowed
    def prepare(count):
        result = _typed_buffers(count)
        trace["typed_CDR"]["owned_typed_buffer_bytes"] += sum(array.nbytes for array in result[:3])
        return result
    def decode(cdr, count):
        trace["typed_CDR"]["pose_calls" if count == 7 else "covariance_calls"] += 1
        return _typed_decoder(cdr, count)
    def finish(stamps, poses, covariance, seen):
        _typed_complete(stamps, poses, covariance, seen)
        trace["typed_CDR"]["complete_records"] += seen
    first = '    stamps, poses, covariance, frames = [], [], [], set()'
    last = '    timings["raw_pose_and_calibration_seconds"] = time.perf_counter()-before'
    if source.count(first) != 1 or source.count(last) != 1:
        raise ValueError("The original complete pose and calibration block changed")
    begin, end = source.index(first), source.index(last)+len(last)
    native_block = source[begin:end]
    replacements = [
        (first, '    stamps, poses, covariance, frames = _typed_prepare(_trajectory_count)\\n    odometry_seen = 0'),
        ('    for message in index.iter_messages("/localization/pose/odom"):',
         '    for odometry_index, message in enumerate(index.iter_messages("/localization/pose/odom")):'),
        ('        stamps.append(sec*1_000_000_000+ns)', '        stamps[odometry_index] = sec*1_000_000_000+ns'),
        ('        poses.append([cdr.f64() for _ in range(7)])', '        poses[odometry_index] = _typed_decode(cdr, 7)'),
        ('        covariance.append([cdr.f64() for _ in range(36)])',
         '        covariance[odometry_index] = _typed_decode(cdr, 36)\\n        odometry_seen = odometry_index+1'),
        ('    if frames != {("odom", "base_link")}:',
         '    _typed_finish(stamps, poses, covariance, odometry_seen)\\n    if frames != {("odom", "base_link")}:')]
    typed_block = native_block
    for before, value in replacements:
        if typed_block.count(before) != 1:
            raise ValueError("A native complete trajectory field changed its unique call site")
        typed_block = typed_block.replace(before, value)
    combined = ('    _trajectory_count = index.message_count("/localization/pose/odom")\\n'
        '    if _typed_admit(_trajectory_count):\\n' + textwrap.indent(typed_block, '    ') + '\\n'
        '    else:\\n' + textwrap.indent(native_block, '    '))
    source = source[:begin]+combined+source[end:]
    namespace.update(_effective_pairs=pairs, _effective_exports=exports, _typed_admit=admit,
        _typed_prepare=prepare, _typed_decode=decode, _typed_finish=finish)'''
    if factory.count(anchor) != 1:
        raise ValueError("The accepted raw route no longer has its single composition binding")
    namespace = dict(run_effective_initial.__dict__)
    import textwrap
    namespace.update(textwrap=textwrap, _typed_admission=admitted, _typed_buffers=buffers,
        _typed_decoder=decode_view, _typed_complete=complete, _typed_maximum_records=MAXIMUM_RECORDS)
    if enabled:
        exec(compile(factory.replace(anchor, after), "<accepted_raw_route_with_typed_native_trajectory>", "exec"), namespace)
        run_effective_initial.combined_raw_execution = namespace["combined_raw_execution"]
    try:
        yield
    finally:
        run_effective_initial.combined_raw_execution = original
