"""Compare native scalar CDR values and typed record ownership without object priors."""

import argparse
import importlib
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"),
               str(ROOT/"02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
from typed_odometry_cdr import buffers, complete, admitted, decode_view, MAXIMUM_RECORDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import Cdr
    checks = {}
    for count in (7, 36):
        values = [-0., 1., -3.125, 1e-100, 1e100, float("inf"), float("nan")]+[float(index) for index in range(count-7)]
        for offset in range(4, 20):
            start = offset+(-(offset-4)) % 8
            raw = bytes(start)+struct.pack("<"+str(count)+"d", *values)+b"tail"
            native, typed = Cdr(raw), Cdr(raw)
            native.offset = typed.offset = offset
            a, b = [native.f64() for _ in range(count)], decode_view(typed, count)
            checks[f"count{count}_offset{offset}_all_bits_and_final_offset_exact"] = struct.pack("<"+str(count)+"d", *a) == b.tobytes() and native.offset == typed.offset
            truncated = Cdr(raw[:start+8*count-1])
            truncated.offset = offset
            try:
                decode_view(truncated, count)
            except struct.error:
                checks[f"count{count}_offset{offset}_truncated_rejected"] = True
            else:
                checks[f"count{count}_offset{offset}_truncated_rejected"] = False
    for records in (0, 1, 2, 19):
        stamps, poses, covariance, frames = buffers(records)
        for index in range(records):
            stamps[index] = 1700000000000000000+index*1000000
            poses[index] = np.array([index*.01, index*.02, -index*.03, 0., 0., 0., 1.])
            covariance[index] = np.arange(36, dtype=np.float64)*(-1 if index % 2 else 0)
        complete(stamps, poses, covariance, records)
        checks[f"records{records}_owned_complete_covariance_dtype_shape_and_strides"] = all(a.flags.owndata and a.flags.c_contiguous for a in (stamps, poses, covariance)) and stamps.dtype == np.int64 and poses.dtype == covariance.dtype == np.float64
        if records:
            try:
                complete(stamps, poses, covariance, records-1)
            except ValueError:
                checks[f"records{records}_partial_fill_rejected"] = True
            else:
                checks[f"records{records}_partial_fill_rejected"] = False
    checks["unsupported_size_selects_native_before_allocation"] = not admitted(MAXIMUM_RECORDS+1) and not admitted(-1) and not admitted(1.)
    covariance = buffers(2)[2]
    covariance[:] = -0.
    all_zero = bool(np.all(covariance == 0))
    covariance[1, 11] = float("nan")
    checks["all36_nonzero_nan_and_signed_zero_covariance_semantics_preserved"] = all_zero and not bool(np.all(covariance == 0))
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("typed_odometry_cdr.py"))],
        "counted": False, "no_GT_SfM_shape_or_fixed_scene_frame_count_input": True}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": len(checks)}, flush=True)
    if not all(checks.values()):
        raise ValueError("A native trajectory typed representation guard failed")


if __name__ == "__main__":
    main()
