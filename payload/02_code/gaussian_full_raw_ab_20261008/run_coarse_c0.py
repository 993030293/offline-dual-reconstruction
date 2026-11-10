"""Generate coarse camera-support motion from raw LiDAR and IMU without SfM."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/code/reference_sources/c0_lio_fallback/tools"
sys.path[:0] = [str(TOOLS), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]
from runtime import atomic_json, data_identity
import lio_local_map as native
from mcap_audit import McapIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-floor-constraint", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    index = McapIndex(args.raw)
    total = index.message_count(native.TOPIC)
    from lio_baseline import load_imu
    imu_ns, gyro, accel = load_imu(index)
    seconds = (imu_ns - imu_ns[0]) * 1e-9
    interval = (seconds >= .5) & (seconds <= 11.5)
    if interval.sum() < 100 or not np.isfinite(gyro).all() or not np.isfinite(accel).all():
        raise ValueError("The required raw IMU bias interval has insufficient valid support")
    bias = gyro[interval].mean(0)
    diagnostic = {"start_s": .5, "end_s": 11.5, "sample_count": int(interval.sum()),
        "gyro_bias_rad_s": bias.tolist(), "gyro_std_rad_s": gyro[interval].std(0).tolist(),
        "gyro_residual_norm_p95_rad_s": float(np.quantile(np.linalg.norm(gyro[interval] - bias, axis=1), .95)),
        "acceleration_mean_m_s2": accel[interval].mean(0).tolist(),
        "acceleration_std_m_s2": accel[interval].std(0).tolist(),
        "IMU_alone_cannot_confirm_zero_translation": True,
        "static_hold_status": "retained_algorithm_assumption_not_independently_confirmed"}
    atomic_json(args.output / "INITIAL_IMU_DIAGNOSTIC.json", diagnostic)
    state = {"status": "raw_LiDAR_IMU_running", "pid": os.getpid(), "raw": str(args.raw),
             "completed_scans": 0, "total_scans": total}
    atomic_json(args.output / "CONTRACT.json", {"raw": str(args.raw.resolve()), "raw_bytes": args.raw.stat().st_size,
        "algorithm": "retained_IMU_rotation_and_point_to_plane_translation_local_map_ICP",
        "official_Point_LIO": False, "accelerometer_propagation": False, "loop_closure": False,
        "SfM_or_endpoint_visual_rotation_constraint": False, "full_sequence_requested": True,
        "voxel_m": .15, "max_points_per_scan": 2500, "max_map_points": 10000,
        "map_frames": 8, "correspondence_m": .45, "iterations": 8, "initial_static_hold_seconds": 11.5,
        "initial_static_hold_requires_sensor_check": True, "unit": "meters", "GT_used": False,
        "sources": [data_identity(Path(__file__)), data_identity(Path(native.__file__)),
                    data_identity(TOOLS / "lio_baseline.py"), data_identity(TOOLS / "mcap_audit.py")]})
    original = index.iter_messages

    def messages(topic, *parameters, **keywords):
        for number, row in enumerate(original(topic, *parameters, **keywords), 1):
            if topic == native.TOPIC and (number % 25 == 0 or number == total):
                state.update(completed_scans=number, elapsed_seconds=time.perf_counter() - started,
                             rss_bytes=psutil.Process().memory_info().rss)
                atomic_json(args.output / "STATE.json", state)
                print(json.dumps(state), flush=True)
            yield row

    index.iter_messages = messages
    floor_constraint = None
    if args.measured_floor_constraint:
        sys.path.insert(0, str(ROOT / '02_code/reconstruction_dual_v1_20261009'))
        from floor_motion_constraint import FloorMotionConstraint, PROTOCOL
        floor_constraint = FloorMotionConstraint()
    constraint_arguments = {'motion_constraint': floor_constraint} if floor_constraint is not None else {}
    report, series = native.run(index, .15, 2500, 10000, 8, .45, 8, 11.5, None, None, True,
                               **constraint_arguments)
    poses = series["poses"]
    if (len(poses) != total or not np.isfinite(poses).all()
            or np.any(np.diff(series["timestamps_ns"]) <= 0)):
        raise ValueError("Full raw motion output has invalid coverage, pose or clock")
    extra = {}
    if floor_constraint is not None:
        floor_receipt = floor_constraint.receipt()
        if floor_receipt['actual_scan_count'] != total:
            raise ValueError('Measured floor correction clocks do not cover the original raw scans')
        floor_path = args.output / 'MEASURED_FLOOR_RECEIPT.json'
        atomic_json(floor_path, floor_receipt)
        extra = {'gyro_world_correction_xyzw': np.asarray(floor_constraint.corrections),
                 'measured_floor_supported': np.asarray(floor_constraint.supported, dtype=bool),
                 'measured_floor_motion_accepted': np.asarray([row['motion_accepted']
                     for row in floor_constraint.rows], dtype=bool)}
        if not np.array_equal(extra['measured_floor_motion_accepted'], series['accepted']):
            raise ValueError('Raw motion and measured plane acceptance differ')
        report.update(measured_floor_constraint_protocol=PROTOCOL,
            measured_floor_receipt=str(floor_path), measured_floor_constraint_used=True,
            orientation='gyro_prediction_with_same_scan_measured_floor_normal_correction',
            scan_end_clock='full_raw_point_offsets_clipped_to_imu_support',
            floor_reference_is_one_plane_assumption=True,
            dense_registration_requires_saved_per_scan_gyro_world_correction=True)
        contract = json.loads((args.output / 'CONTRACT.json').read_text(encoding='utf-8'))
        contract.update(algorithm='raw_gyro_measured_floor_and_point_to_plane_local_map',
            measured_floor_constraint_protocol=PROTOCOL,
            original_floor_quality_rules_preserved=True,
            dense_registration_requires_saved_per_scan_gyro_world_correction=True,
            floor_reference_is_one_plane_assumption=True)
        contract['sources'] += [data_identity(ROOT / '02_code/reconstruction_dual_v1_20261009/floor_motion_constraint.py'),
                                data_identity(ROOT / '05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/code/estimate_sparse_floor.py')]
        atomic_json(args.output / 'CONTRACT.json', contract)
    np.savez(args.output / "coarse_body_odometry.npz", timestamp_ns=series["timestamps_ns"],
        pose_xyz_xyzw=np.column_stack((poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat())),
        accepted=series["accepted"], **extra)
    np.savez(args.output / "coarse_sensor_diagnostics.npz", timestamp_ns=series["timestamps_ns"],
        rmse=series["rmse"], correspondences=series["correspondences"], accepted=series["accepted"])
    report.update(raw_file=str(args.raw.resolve()), elapsed_seconds=time.perf_counter() - started,
                  official_Point_LIO=False, SfM_used=False, GT_used=False, full_input_scan_count=total,
                  initial_static_hold_status=diagnostic["static_hold_status"],
                  initial_static_hold_seconds=11.5,
                  status="full_raw_coarse_motion_completed_geometry_accuracy_unverified")
    atomic_json(args.output / "COARSE_RECEIPT.json", report)
    state.update(status=report["status"], completed_scans=total, elapsed_seconds=report["elapsed_seconds"])
    atomic_json(args.output / "STATE.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
