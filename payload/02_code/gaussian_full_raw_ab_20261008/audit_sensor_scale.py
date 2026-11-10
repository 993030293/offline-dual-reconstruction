"""Report raw LiDAR consistency in estimated metric coordinates, never ground-truth accuracy."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pycolmap
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/code/reference_sources/c0_lio_fallback/tools"
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"), str(TOOLS)]
from runtime import atomic_json, data_identity
from lio_baseline import decode_points, voxel_select, LIDAR_TOPIC
import run_initial as initial


def static_transform(rows, target):
    graph = {}
    registered = {}
    for row in rows:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(row["rotation_xyzw"]).as_matrix()
        matrix[:3, 3] = row["translation"]
        edge = row["parent"], row["child"]
        if edge in registered and not np.array_equal(registered[edge], matrix):
            raise ValueError("A raw LiDAR extrinsic changed within the recording")
        registered[edge] = matrix
    for (parent, child), matrix in registered.items():
        graph.setdefault(parent, []).append((child, matrix))
        graph.setdefault(child, []).append((parent, np.linalg.inv(matrix)))
    pending = [("base_link", np.eye(4), ["base_link"])]
    seen = set()
    while pending:
        frame, matrix, path = pending.pop(0)
        if frame == target:
            return matrix, path
        if frame in seen:
            continue
        seen.add(frame)
        for child, value in graph.get(frame, []):
            if child not in seen:
                pending.append((child, matrix @ value, path+[child]))
    raise ValueError("The raw LiDAR frame has no verified body extrinsic path")


def distribution(values):
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Consistency distances require a complete finite denominator")
    quantiles = np.quantile(values, [.5, .9, .95, .99])
    return {"count": len(values), "p50_m": float(quantiles[0]), "p90_m": float(quantiles[1]),
        "p95_m": float(quantiles[2]), "p99_m": float(quantiles[3]), "max_m": float(values.max()),
        "fractions_within_m": {str(value): float(np.mean(values <= value)) for value in (.02, .05, .1, .2, .5)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    admission = json.loads((args.side / "export/TRAINING_ADMISSION.json").read_text())
    if admission["status"] != "export_and_admission_verified_training_pending":
        raise ValueError("Metric consistency uses only the admitted fresh map")
    model_path = Path(admission["training_opencv_model"])
    motion_path = args.side / "initial/raw_body_odometry.npz"
    sources = [data_identity(motion_path)]+[data_identity(path) for path in sorted(model_path.glob("*.bin"))]
    raw_identity = {"path": str(args.raw.resolve()), "bytes": args.raw.stat().st_size, "mtime_ns": args.raw.stat().st_mtime_ns}
    index = initial.McapIndex(args.raw)
    scan_count = index.message_count(LIDAR_TOPIC)
    selected = sorted({int((number+.5)*scan_count/64) for number in range(64)})
    rows = []
    transforms = []
    for message in index.iter_messages("/tf_static"):
        transforms.extend(initial.decode_tf_static(message["payload"]))
    with np.load(motion_path) as saved:
        stamps, poses = saved["timestamp_ns"].copy(), saved["pose_xyz_xyzw"].copy()
    contract = {"status": "frozen_before_consistency", "raw": raw_identity, "sources": sources,
        "source": data_identity(Path(__file__)), "selected_raw_scan_ids": selected,
        "scan_selection": "64 raw-index stratum midpoints independent of both map outputs",
        "raw_valid_range_m": [.15, 30.], "scan_voxel_m": .05, "maximum_points_per_scan": 5000,
        "per_point_raw_time_offset_used": True, "maximum_motion_bracket_ns": 120000000,
        "offline_future_bracketing": True, "estimated_motion_not_GT": True,
        "independent_dimensions_or_ATE_RPE_available": False, "object_ROI_or_box_dimensions_used": False,
        "distance_is_density_dependent_sensor_consistency_not_surface_error": True}
    atomic_json(args.output / "CONTRACT.json", contract)
    world_blocks, sensor_frame = [], None
    curve = None
    for scan_id, message in index.iter_selected_messages(LIDAR_TOPIC, selected):
        _, _, frame = initial.Cdr(message["payload"]).header()
        if sensor_frame is not None and sensor_frame != frame:
            raise ValueError("The LiDAR frame changes within the complete diagnostic denominator")
        sensor_frame = frame
        if curve is None:
            matrix, path = static_transform(transforms, frame)
            curve = initial.baseline.CoarseCameraCurve(stamps, poses, matrix)
            contract.update(sensor_frame=frame, T_body_lidar=matrix.tolist(), extrinsic_path=path)
            atomic_json(args.output / "CONTRACT.json", contract)
        stamp, points, offsets = decode_points(message["payload"])
        points, offsets = voxel_select(points, offsets, .05, 5000)
        motion = initial.coarse_camera_at_many(curve, stamp+offsets, 120000000)
        valid = np.asarray([row["coarse_pose_valid"] for row in motion], bool)
        supported = [row for row in motion if row["coarse_pose_valid"]]
        if supported:
            rotations = np.asarray([row["R_WC"] for row in supported])
            centers = np.asarray([row["C_W_m"] for row in supported])
            world = np.einsum("nij,nj->ni", rotations, points[valid])+centers
            if not np.isfinite(world).all():
                raise ValueError("Estimated metric LiDAR positions contain invalid values")
            world_blocks.append(world)
        rows.append({"raw_scan_id": scan_id, "header_ns": stamp, "valid_downsampled_points": len(points),
            "motion_supported_points": int(valid.sum()), "unsupported_points_retained_in_denominator": int((~valid).sum())})
    if len(rows) != len(selected) or not world_blocks:
        raise ValueError("The raw scan diagnostic has incomplete coverage")
    lidar = np.concatenate(world_blocks)
    # Identical native voxel representatives are used on either independently estimated trajectory.
    lidar, _ = voxel_select(lidar, np.zeros(len(lidar), np.int64), .05, len(lidar))
    model = pycolmap.Reconstruction(model_path)
    point_ids = sorted(model.points3D)
    sparse = np.asarray([model.points3D[identity].xyz for identity in point_ids])
    sparse_to_lidar = cKDTree(lidar).query(sparse, workers=1)[0]
    lidar_to_sparse = cKDTree(sparse).query(lidar, workers=1)[0]
    np.savez(args.output / "CONSISTENCY_ARRAYS.npz", lidar_world_m=lidar, map_point_ids=np.asarray(point_ids, np.int64),
             sparse_to_lidar_distance_m=sparse_to_lidar, lidar_to_sparse_distance_m=lidar_to_sparse)
    if sources != [data_identity(motion_path)]+[data_identity(path) for path in sorted(model_path.glob("*.bin"))]:
        raise ValueError("Read-only sensor diagnostics changed a source map or estimated motion")
    result = {"status": "raw_metric_sensor_consistency_completed_accuracy_unverified", "raw_scan_denominator": scan_count,
        "selected_scan_denominator": len(selected), "rows": rows, "compared_unique_lidar_points": len(lidar),
        "compared_sparse_points": len(sparse), "sparse_to_raw_LiDAR": distribution(sparse_to_lidar),
        "raw_LiDAR_to_sparse_map": distribution(lidar_to_sparse), "elapsed_seconds": time.monotonic()-started,
        "all_selected_scan_failures_and_unsupported_points_retained": True, "source_unchanged": True,
        "GT_or_absolute_accuracy_claim": False, "scale_unit": "estimated_motion_and_raw_LiDAR_meters",
        "confounds": ["rough_motion_error", "LiDAR_camera_calibration", "dynamic_surfaces", "sampling_and_sparse_point_density"]}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "sparse_to_raw_LiDAR")}))


if __name__ == "__main__":
    main()
