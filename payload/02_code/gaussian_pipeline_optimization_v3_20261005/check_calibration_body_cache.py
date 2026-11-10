"""Check every real calibration message and generic native CDR/cache boundaries."""

import argparse
import importlib
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"),
               str(ROOT/"02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
from calibration_body_cache import CalibrationBodyCache, calibration_body_execution
import run_effective_initial


def exact(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is float:
        return struct.pack("<d", a) == struct.pack("<d", b)
    if type(a) is dict:
        return a.keys() == b.keys() and all(exact(a[key], b[key]) for key in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def message(index, *, frame=b"camera_optical", height=620, width=812, model="plumb_bob", distortion=None, matrix=None, tail=b"native_ignored_footer"):
    data = bytearray(b"\x00\x01\x00\x00")
    distortion = [0.01, -0.02, -0., 0.003, 0.004] if distortion is None else distortion
    matrix = [700., 0., 400., 0., 701., 310., 0., 0., 1.] if matrix is None else matrix

    def align(size):
        data.extend(bytes((-(len(data)-4)) % size))

    def scalar(value, fmt):
        align(struct.calcsize(fmt))
        data.extend(struct.pack("<"+fmt, value))

    def string(value):
        raw = value.encode() if isinstance(value, str) else value
        scalar(len(raw)+1, "I")
        data.extend(raw+b"\x00")

    scalar(1700000000+index//1000000000, "i")
    scalar(index % 1000000000, "I")
    string(frame)
    scalar(height, "I")
    scalar(width, "I")
    string(model)
    scalar(len(distortion), "I")
    for value in distortion+matrix:
        scalar(value, "d")
    data.extend(tail)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex, Cdr, decode_camera_info
    paths = [Path(__file__), Path(__file__).with_name("calibration_body_cache.py"), baseline.TOOLS/"mcap_audit.py"]
    sources = [data_identity(p) for p in paths]
    native_stat = baseline.RAW.stat()
    index = McapIndex(baseline.RAW)
    cache = CalibrationBodyCache(decode_camera_info, Cdr, True)
    count, real_exact = 0, True
    for row in index.iter_messages("/sensing/camera/camera_info"):
        real_exact = real_exact and exact(decode_camera_info(row["payload"]), cache(row["payload"]))
        count += 1
    checks = {"all7338_real_complete_native_fields_and_headers_bit_exact": real_exact and count == index.message_count("/sensing/camera/camera_info") == 7338,
        "original_raw_source_size_and_mtime_preserved": (native_stat.st_size, native_stat.st_mtime_ns)
            == (baseline.RAW.stat().st_size, baseline.RAW.stat().st_mtime_ns)}
    generic = CalibrationBodyCache(decode_camera_info, Cdr, True)
    payloads = [message(i, frame=frame, height=h, width=w, model=model, distortion=d)
        for i in range(3) for frame in (b"cam", b"different_optical_frame", b"bad\xffframe")
        for h, w, model, d in ((1, 1, "", []), (512, 1024, "equidistant", [.1, -.2, .3, -.4]),
                               (1937, 487, "plumb_bob", [0.]*8))]
    checks["all27_multisize_multimodel_alignment_and_header_variants_exact"] = all(exact(decode_camera_info(p), generic(p)) for p in payloads)
    mutable = CalibrationBodyCache(decode_camera_info, Cdr, True)
    first, second = message(1), message(2)
    result = mutable(first)
    result["camera_matrix"][0] = -7.
    result["distortion"].append(33.)
    result["frame_id"] = "mutated"
    restored = mutable(second)
    checks["caller_mutation_does_not_change_private_cache_or_new_header"] = exact(decode_camera_info(second), restored) and mutable.trace["hits"] == 1
    restored["camera_matrix"][0] = -8.
    checks["hit_returns_are_independent_owned_lists"] = exact(decode_camera_info(first), mutable(first))
    changed = message(3, matrix=[701., 0., 400., 0., 701., 310., 0., 0., 1.])
    changed_info = mutable(changed)
    checks["changed_K_and_tail_not_reused_or_hidden"] = (exact(decode_camera_info(changed), changed_info)
        and changed_info["camera_matrix"] != decode_camera_info(first)["camera_matrix"]
        and exact(decode_camera_info(message(4, tail=b"different_footer")), mutable(message(4, tail=b"different_footer"))))
    nonfinite = CalibrationBodyCache(decode_camera_info, Cdr, True)
    bad = message(0, distortion=[float("nan"), float("inf"), -0.])
    nonfinite_a, nonfinite_b = nonfinite(bad), nonfinite(bad)
    checks["nonfinite_original_values_uncached_and_NaN_identity_comparison_preserved"] = exact(nonfinite_a, decode_camera_info(bad)) and exact(nonfinite_b, decode_camera_info(bad)) and nonfinite.trace["hits"] == 0 and nonfinite_a["distortion"] != nonfinite_b["distortion"]
    one = CalibrationBodyCache(decode_camera_info, Cdr, True, capacity=1)
    one(first)
    one(changed)
    checks["capacity_admission_selects_original_without_eviction_or_record_loss"] = exact(one(changed), decode_camera_info(changed)) and len(one.entries) == 1 and one.trace["native_calls"] == 3
    small = CalibrationBodyCache(decode_camera_info, Cdr, True, maximum_payload_bytes=64)
    checks["oversized_and_bytearray_inputs_preselect_native"] = exact(small(first), decode_camera_info(first)) and exact(small(bytearray(first)), decode_camera_info(bytearray(first))) and not small.entries
    error_checks = []
    for broken in (b"", first[:8], first[:36], first[:-len(b"native_ignored_footer")-1]):
        observed = []
        for decoder in (decode_camera_info, CalibrationBodyCache(decode_camera_info, Cdr, True)):
            try:
                decoder(broken)
            except (ValueError, struct.error, IndexError) as error:
                observed.append((type(error).__name__, str(error)))
            else:
                observed.append(None)
        error_checks.append(observed[0] == observed[1] and observed[0] is not None)
    checks["all_original_truncation_errors_propagated_exactly"] = all(error_checks)
    original = run_effective_initial.combined_raw_execution
    try:
        with calibration_body_execution(True):
            raise ValueError("owned_boundary_error")
    except ValueError:
        pass
    checks["global_factory_restored_after_error"] = run_effective_initial.combined_raw_execution is original
    checks["sources_unchanged"] = sources == [data_identity(p) for p in paths]
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "actual_count": count, "real_trace": cache.trace, "generic_trace": generic.trace,
        "sources": sources, "counted": False, "multi_calibration_full_pipeline_not_claimed": True}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": checks, "real_trace": cache.trace}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original CameraInfo field, header, mutation or failure contract changed")


if __name__ == "__main__":
    main()
