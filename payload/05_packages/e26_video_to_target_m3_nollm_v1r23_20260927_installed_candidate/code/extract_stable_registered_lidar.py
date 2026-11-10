#!/usr/bin/env python3
"""Extract temporally paired, multi-scan-stable registered LiDAR anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TOPIC = "/localization/lidar/points_registered_full"
MINIMUM_TOTAL_ANCHORS = 10000
MINIMUM_STRONG_FRAME_ANCHORS = 10
MAXIMUM_SPARSE_FRAME_FRACTION = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def decode_xyz(message) -> np.ndarray:
    fields = {field.name: field for field in message.fields}
    for name in ("x", "y", "z"):
        if name not in fields or int(fields[name].datatype) != 7 or int(fields[name].count) != 1:
            raise RuntimeError(f"PointCloud2 field {name} is not scalar FLOAT32")
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [int(fields[name].offset) for name in ("x", "y", "z")],
            "itemsize": int(message.point_step),
        }
    )
    values = np.frombuffer(message.data, dtype=dtype, count=int(message.width) * int(message.height))
    return np.column_stack((values["x"], values["y"], values["z"])).astype(np.float32)


def assess_anchor_coverage(rows: list[dict]) -> dict:
    if len(rows) < 3:
        raise RuntimeError("Too few frames for stable LiDAR coverage assessment")
    counts = np.asarray([int(row["stable_anchor_point_count"]) for row in rows])
    total = int(counts.sum())
    if total < MINIMUM_TOTAL_ANCHORS:
        raise RuntimeError(f"Stable LiDAR anchors are insufficient: total={total}")
    sparse_indices = np.flatnonzero(counts < MINIMUM_STRONG_FRAME_ANCHORS).tolist()
    maximum_sparse = max(1, int(np.floor(MAXIMUM_SPARSE_FRAME_FRACTION * len(rows))))
    if len(sparse_indices) > maximum_sparse:
        raise RuntimeError(
            "Stable LiDAR anchors have too many sparse frames: "
            f"{len(sparse_indices)}/{len(rows)}, allowed={maximum_sparse}"
        )
    sparse_set = set(sparse_indices)
    for index in sparse_indices:
        if (
            counts[index] == 0
            or index == 0
            or index == len(rows) - 1
            or index - 1 in sparse_set
            or index + 1 in sparse_set
        ):
            raise RuntimeError(
                "Stable LiDAR anchors have an unsupported sparse frame: "
                f"index={index}, count={counts[index]}"
            )
    return {
        "status": "isolated_sparse_frames" if sparse_indices else "strong_all_frames",
        "minimum_total_anchors": MINIMUM_TOTAL_ANCHORS,
        "minimum_strong_frame_anchors": MINIMUM_STRONG_FRAME_ANCHORS,
        "maximum_sparse_frame_fraction": MAXIMUM_SPARSE_FRAME_FRACTION,
        "maximum_sparse_frame_count": maximum_sparse,
        "sparse_frame_count": len(sparse_indices),
        "sparse_frames": [
            {
                "image_name": rows[index]["image_name"],
                "frame_index": index,
                "cropped_non_ground_point_count": rows[index]["cropped_non_ground_point_count"],
                "stable_anchor_point_count": int(counts[index]),
            }
            for index in sparse_indices
        ],
        "minimum_frame_anchor_count": int(counts.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--roi-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-pair-delta-ms", type=float, default=60.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.02)
    parser.add_argument("--minimum-scan-support", type=int, default=3)
    parser.add_argument("--minimum-ground-clearance-m", type=float, default=0.03)
    args = parser.parse_args()

    from rosbags.highlevel import AnyReader

    mcap = args.mcap.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    roi_path = args.roi_receipt.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scans_dir = output / "scans"
    scans_dir.mkdir()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    roi_receipt = json.loads(roi_path.read_text(encoding="utf-8-sig"))
    image_times = np.asarray([int(row["camera_header_ns"]) for row in manifest], dtype=np.int64)
    image_names = [str(row["image_name"]) for row in manifest]
    roi_minimum = np.asarray(roi_receipt["roi_world_lio_m"]["minimum"], dtype=np.float64)
    roi_maximum = np.asarray(roi_receipt["roi_world_lio_m"]["maximum"], dtype=np.float64)
    support = roi_receipt["roi_derivation"]["metric_sparse_vehicle_support"]
    ground_z = float(support["ground_z_m"])
    best_delta = np.full(len(image_times), np.iinfo(np.int64).max, dtype=np.int64)
    best_stamp = np.full(len(image_times), -1, dtype=np.int64)
    best_points: list[np.ndarray | None] = [None] * len(image_times)

    with AnyReader([mcap]) as reader:
        connections = [item for item in reader.connections if item.topic == TOPIC]
        if len(connections) != 1:
            raise RuntimeError(f"Expected one {TOPIC} connection, found {len(connections)}")
        for connection, _log_time, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            if str(message.header.frame_id).lstrip("/") != "odom":
                raise RuntimeError(f"Registered LiDAR frame is not odom: {message.header.frame_id}")
            stamp = timestamp_ns(message.header.stamp)
            insertion = int(np.searchsorted(image_times, stamp))
            candidate_indices = []
            if insertion < len(image_times):
                candidate_indices.append(insertion)
            if insertion > 0:
                candidate_indices.append(insertion - 1)
            if not candidate_indices:
                continue
            improved = [
                index
                for index in candidate_indices
                if abs(stamp - int(image_times[index])) < int(best_delta[index])
            ]
            if not improved:
                continue
            points = decode_xyz(message)
            valid = np.isfinite(points).all(axis=1)
            valid &= np.all(points >= roi_minimum[None, :], axis=1)
            valid &= np.all(points <= roi_maximum[None, :], axis=1)
            valid &= points[:, 2] > ground_z + args.minimum_ground_clearance_m
            cropped = np.ascontiguousarray(points[valid])
            for index in improved:
                delta = abs(stamp - int(image_times[index]))
                best_delta[index] = delta
                best_stamp[index] = stamp
                best_points[index] = cropped

    maximum_delta_ns = int(round(args.maximum_pair_delta_ms * 1e6))
    if np.any(best_delta > maximum_delta_ns) or any(points is None for points in best_points):
        failures = [
            {"image": image_names[index], "delta_ms": float(best_delta[index] / 1e6)}
            for index in range(len(image_names))
            if best_points[index] is None or best_delta[index] > maximum_delta_ns
        ]
        raise RuntimeError(f"Registered LiDAR pairing failed: {failures[:10]}")

    unique_scans: dict[int, np.ndarray] = {}
    for stamp, points in zip(best_stamp.tolist(), best_points):
        if stamp not in unique_scans:
            unique_scans[stamp] = points
    voxel_rows = []
    scan_stamps = sorted(unique_scans)
    for scan_index, stamp in enumerate(scan_stamps):
        points = unique_scans[stamp]
        voxels = np.floor(points / args.voxel_size_m).astype(np.int32)
        voxels = np.unique(voxels, axis=0)
        voxel_rows.append(
            np.column_stack((voxels, np.full(len(voxels), scan_index, dtype=np.int32)))
        )
    voxel_scan_pairs = np.unique(np.concatenate(voxel_rows, axis=0), axis=0)
    stable_voxels, scan_support = np.unique(voxel_scan_pairs[:, :3], axis=0, return_counts=True)
    keep_voxels = stable_voxels[scan_support >= args.minimum_scan_support]
    keep_dtype = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
    keep_keys = np.ascontiguousarray(keep_voxels).view(keep_dtype).reshape(-1)

    rows = []
    total_input_points = 0
    total_stable_points = 0
    for index, (name, stamp, delta, points) in enumerate(
        zip(image_names, best_stamp.tolist(), best_delta.tolist(), best_points)
    ):
        voxels = np.floor(points / args.voxel_size_m).astype(np.int32)
        keys = np.ascontiguousarray(voxels).view(keep_dtype).reshape(-1)
        stable = np.isin(keys, keep_keys)
        retained = np.ascontiguousarray(points[stable])
        output_path = scans_dir / f"{Path(name).stem}.npz"
        np.savez_compressed(
            output_path,
            points_world_lio_m=retained,
            image_timestamp_ns=np.asarray(image_times[index], dtype=np.int64),
            lidar_timestamp_ns=np.asarray(stamp, dtype=np.int64),
        )
        total_input_points += len(points)
        total_stable_points += len(retained)
        rows.append(
            {
                "image_name": name,
                "image_timestamp_ns": int(image_times[index]),
                "lidar_timestamp_ns": int(stamp),
                "absolute_delta_ms": float(delta / 1e6),
                "cropped_non_ground_point_count": len(points),
                "stable_anchor_point_count": len(retained),
                "path": output_path.relative_to(output).as_posix(),
            }
        )
    anchor_coverage = assess_anchor_coverage(rows)
    receipt = {
        "schema_version": "e26_stable_registered_lidar_anchors_v2",
        "status": "complete",
        "source_mcap": str(mcap),
        "topic": TOPIC,
        "source_frame": "odom",
        "target_geometry_frame": "world_lio_same_as_odom_metric_context",
        "image_manifest": str(manifest_path),
        "image_manifest_sha256": sha256_file(manifest_path),
        "roi_receipt": str(roi_path),
        "roi_receipt_sha256": sha256_file(roi_path),
        "image_count": len(rows),
        "unique_lidar_scan_count": len(unique_scans),
        "pair_delta_ms": {
            "p50": float(np.quantile(best_delta / 1e6, 0.50)),
            "p95": float(np.quantile(best_delta / 1e6, 0.95)),
            "maximum": float(np.max(best_delta / 1e6)),
        },
        "voxel_size_m": args.voxel_size_m,
        "minimum_scan_support": args.minimum_scan_support,
        "minimum_ground_clearance_m": args.minimum_ground_clearance_m,
        "ground_z_m": ground_z,
        "cropped_non_ground_point_count": total_input_points,
        "stable_anchor_point_count": total_stable_points,
        "stable_voxel_count": len(keep_voxels),
        "anchor_coverage": anchor_coverage,
        "per_frame": rows,
        "language_model_used": False,
        "semantic_model_used": False,
        "manual_selection_used": False,
        "scientific_boundary": (
            "Anchors are raw registered LiDAR returns retained by metric ROI, ground clearance, "
            "timestamp pairing, and repeated 2 cm voxel support. Isolated sparse frames "
            "retain only their observed anchors; missing glass returns remain unknown."
        ),
    }
    receipt_path = output / "stable_lidar_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
