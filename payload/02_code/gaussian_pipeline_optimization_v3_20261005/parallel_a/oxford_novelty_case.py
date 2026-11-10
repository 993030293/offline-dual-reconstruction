"""Measure all raw Oxford body-pose novelty queries without GT or image selection."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from spatial_novelty import NoveltyIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "hash"), required=True)
    for name in ("input", "source-receipt", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    sources = [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("spatial_novelty.py"), args.input, args.source_receipt)]
    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    if (receipt["status"] != "RAW_LIO_ESTIMATED_FULL_SEQUENCE" or receipt["units"] != "metric_m"
            or receipt["quaternion_order"] != "xyzw" or set(receipt["source_roles"]) != {"raw_measurement", "sensor_calibration"}
            or any(receipt[key] for key in ("gt_used", "official_trajectory_used", "old_reconstruction_used"))):
        raise ValueError("The public input is not the declared full raw metric LIO estimate")
    with args.input.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["timestamp_ns", "px", "py", "pz", "qx", "qy", "qz", "qw"]:
            raise ValueError("The original complete Oxford pose schema changed")
        rows = list(reader)
    stamps = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    poses = np.asarray([[float(row[key]) for key in ("px", "py", "pz", "qx", "qy", "qz", "qw")] for row in rows])
    if (len(rows) != receipt["estimated_pose_count"] or not np.isfinite(poses).all() or not np.all(np.diff(stamps) > 0)
            or int(stamps[0]) != receipt["start_timestamp_ns"] or int(stamps[-1]) != receipt["end_timestamp_ns"]):
        raise ValueError("Some original full pose records or source epochs changed")
    positions, rotations = poses[:, :3], Rotation.from_quat(poses[:, 3:])
    radius, angle = .035, 1.2
    trace = {key: 0 for key in ("hash_instances", "native_instances", "hash_queries", "native_queries", "selected_rows_considered", "actual_angular_rows")}
    index = NoveltyIndex(positions, rotations, radius, angle, trace) if args.mode == "hash" else None
    selected, decisions = [0], []
    if index is not None:
        index.add(0)
    for i in range(1, len(rows)):
        if index is None:
            chosen = np.asarray(selected, dtype=int)
            distances = np.linalg.norm(positions[i]-positions[chosen], axis=1)
            angles = np.rad2deg((rotations[i].inv()*rotations[chosen]).magnitude())
            value = bool(np.all((distances >= radius) | (angles >= angle)))
        else:
            value = index.novel(i, selected)
            index.add(i)
        decisions.append(value)
        selected.append(i)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output/"DECISIONS.npz", timestamp_ns=stamps, pose_xyz_xyzw=poses, novelty=np.asarray(decisions, dtype=bool))
    if sources != [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("spatial_novelty.py"), args.input, args.source_receipt)]:
        raise ValueError("An original source or raw public pose input changed")
    atomic_json(args.output/"RESULT.json", {"status": "completed", "mode": args.mode, "sources": sources,
        "all_original_pose_count": len(rows), "all_query_count": len(decisions), "complete_component_seconds": time.perf_counter()-started,
        "trace": trace, "radius_m": radius, "angle_degrees": angle, "body_pose_component_only": True,
        "all_history_rows_queried_without_subsampling": True, "no_GT_SfM_image_or_object_category_inputs": True, "counted": False})


if __name__ == "__main__":
    main()
