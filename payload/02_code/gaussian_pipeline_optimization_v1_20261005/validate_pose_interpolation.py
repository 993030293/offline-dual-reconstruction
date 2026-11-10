"""Compare bounded batch interpolation with analytic motion and all raw timestamps."""

import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from pose_interpolation import coarse_camera_at_many
from runtime import atomic_json, data_identity


def scalar(curve, stamps, maximum_span_ns=120_000_000):
    rows = []
    for stamp in stamps:
        try:
            rows.append(dict(curve.at(stamp, maximum_span_ns), coarse_pose_valid=True))
        except ValueError as error:
            if str(error) not in ("outside_coarse_time_support", "coarse_interpolation_gap_too_large"):
                raise
            rows.append({"coarse_pose_valid": False, "pose_rejection": str(error)})
    return rows


def compare(before, after):
    if len(before) != len(after):
        raise AssertionError("Query denominator changed")
    maximum = 0.
    numeric = {"C_W_m", "R_WC", "quaternion_WC_xyzw", "interpolation_alpha"}
    for first, second in zip(before, after):
        if first.keys() != second.keys():
            raise AssertionError("Query metadata keys differ")
        for key, value in first.items():
            if key in numeric:
                difference = float(np.max(np.abs(np.asarray(value)-np.asarray(second[key]))))
                maximum = max(maximum, difference)
                if difference > 1e-10:
                    raise AssertionError("Batch interpolation differs beyond the frozen numerical guard")
            elif value != second[key]:
                raise AssertionError("Query support, index or metadata differs")
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project/"02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    origin = 1_790_000_000_000_000_000
    stamps = np.array([origin, origin+100_000_000, origin+230_000_000], dtype=np.int64)
    rotation = Rotation.from_euler("z", np.array([[0.], [90.], [180.]]), degrees=True)
    poses = np.column_stack([np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]]), rotation.as_quat()])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    transform[:3, 3] = [1., 0., 0.]
    curve = baseline.CoarseCameraCurve(stamps, poses, transform)
    queries = stamps[0]+np.array([50_000_000, -1, 0, 100_000_000, 100_000_001, 230_000_000, 230_000_001, 50_000_000, 1], dtype=np.int64)
    actual = coarse_camera_at_many(curve, queries)
    maximum = compare(scalar(curve, queries), actual)
    expected_rotation = Rotation.from_euler("z", 45, degrees=True).as_matrix() @ transform[:3, :3]
    checks = {"mixed_support_and_duplicate_oracle": [row["coarse_pose_valid"] for row in actual] == [True, False, True, True, False, True, False, True, True],
        "large_epoch_exact_ns_and_metadata": maximum <= 1e-10,
        "analytic_midpoint_center_lever_arm": np.allclose(actual[0]["C_W_m"], [1+2**-.5, 2**-.5, 0], atol=1e-12, rtol=0),
        "analytic_camera_rotation": np.allclose(actual[0]["R_WC"], expected_rotation, atol=1e-12, rtol=0),
        "exact_endpoints_have_zero_span": all(actual[index]["bracket_span_ns"] == 0 for index in [2, 3, 5]),
        "empty_queries": coarse_camera_at_many(curve, []) == [],
        "all_outside_queries": all(not row["coarse_pose_valid"] for row in coarse_camera_at_many(curve, [origin-2, origin+230_000_002]))}
    edge = coarse_camera_at_many(curve, [origin+150_000_000], 130_000_000)[0]
    checks["maximum_gap_inclusive"] = edge["coarse_pose_valid"] and edge["bracket_span_ns"] == 130_000_000
    flipped = poses.copy()
    flipped[1, 3:] *= -1
    flip_curve = baseline.CoarseCameraCurve(stamps, flipped, transform)
    checks["antipodal_quaternion_matches_native"] = compare(scalar(flip_curve, queries), coarse_camera_at_many(flip_curve, queries)) <= 1e-10
    for key, bad in [("float_stamp", [1.5]), ("matrix_stamp", [[origin]]), ("unsigned_overflow", np.array([2**63], dtype=np.uint64))]:
        try:
            coarse_camera_at_many(curve, bad)
        except ValueError:
            checks[key+"_rejected"] = True
        else:
            checks[key+"_rejected"] = False
    try:
        coarse_camera_at_many(curve, [origin], -1)
    except ValueError:
        checks["negative_span_rejected"] = True
    else:
        checks["negative_span_rejected"] = False
    if not all(checks.values()):
        atomic_json(args.output/"SYNTHETIC_FAILURE.json", checks)
        raise AssertionError("Independent temporal interpolation checks failed")
    odometry = np.load(args.input/"raw_body_odometry.npz")
    sensor = json.loads((args.input/"RAW_SENSOR_CONTRACT.json").read_text())
    raw_rows = [json.loads(line) for line in (args.input/"frame_decisions.jsonl").read_text().splitlines()]
    timestamps = np.array([row["camera_header_ns"] for row in raw_rows], dtype=np.int64)
    raw_curve = baseline.CoarseCameraCurve(odometry["timestamp_ns"], odometry["pose_xyz_xyzw"], sensor["T_body_camera"])
    atomic_json(args.output/"CONTRACT.json", {"input": str(args.input), "query_count": len(timestamps),
        "maximum_seconds_per_trial": 300, "trials": 3, "maximum_abs_pose_difference": 1e-10,
        "minimum_median_speed_gain": .1, "sfm_or_GT_input": False,
        "sources": [data_identity(Path(__file__).parent/"pose_interpolation.py"), data_identity(Path(baseline.__file__))],
        "raw_odometry": data_identity(args.input/"raw_body_odometry.npz"), "sensor": data_identity(args.input/"RAW_SENSOR_CONTRACT.json")})
    rows, reference, maximum = [], None, 0.
    for trial in range(3):
        for variant in (["batched", "scalar"] if trial == 1 else ["scalar", "batched"]):
            started = time.perf_counter()
            result = (scalar if variant == "scalar" else coarse_camera_at_many)(raw_curve, timestamps)
            elapsed = time.perf_counter()-started
            if elapsed > 300:
                raise TimeoutError("Interpolation trial exceeded the registered cap")
            if reference is None:
                reference = result
            maximum = max(maximum, compare(reference, result))
            rows.append({"trial": trial, "variant": variant, "seconds": elapsed,
                         "valid_count": sum(row["coarse_pose_valid"] for row in result)})
    medians = {variant: statistics.median(row["seconds"] for row in rows if row["variant"] == variant) for variant in ("scalar", "batched")}
    gain = 1-medians["batched"]/medians["scalar"]
    result = {"status": "numeric_and_timing_passed_integration_pending" if gain >= .1 else "insufficient_stage_gain",
        "checks": checks, "trials": rows, "median_seconds": medians, "stage_reduction_fraction": gain,
        "maximum_abs_pose_difference": maximum, "query_count": len(timestamps),
        "all_support_metadata_and_failure_reasons_equal": True, "counted": False,
        "source": data_identity(Path(__file__)), "adapter": data_identity(Path(__file__).with_name("pose_interpolation.py"))}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
