#!/usr/bin/env python3
"""Build an SfM workspace with deterministic quality and pose-diverse selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frontend(path: Path):
    spec = importlib.util.spec_from_file_location("e26_frozen_frontend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen frontend: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            values = [float(value) for value in line.split()]
            if len(values) != 8:
                raise ValueError("Trajectory rows must contain eight values")
            rows.append(values)
    data = np.asarray(rows, dtype=np.float64)
    if len(data) < 2 or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Trajectory must contain strictly increasing timestamps")
    return data[:, 0] * 1e9, data[:, 1:4], Rotation.from_quat(data[:, 4:8])


def read_extrinsic(path: Path) -> tuple[np.ndarray, Rotation]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    item = payload.get("T_base_link_camera_optical", payload)
    pose = np.asarray(item["pose_xyz_xyzw"], dtype=np.float64)
    if item.get("source_frame") != "camera_optical" or item.get("target_frame") != "base_link":
        raise ValueError("Expected T_base_link_camera_optical")
    return pose[:3], Rotation.from_quat(pose[3:])


def robust_circle_center(points_xy: np.ndarray) -> np.ndarray:
    design = np.column_stack((2.0 * points_xy, np.ones(len(points_xy))))
    target = np.sum(points_xy * points_xy, axis=1)
    weights = np.ones(len(points_xy), dtype=np.float64)
    solution = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(12):
        weighted = np.sqrt(weights)[:, None]
        solution = np.linalg.lstsq(weighted * design, weighted[:, 0] * target, rcond=None)[0]
        center = solution[:2]
        radii = np.linalg.norm(points_xy - center, axis=1)
        residual = radii - np.median(radii)
        scale = max(1e-6, 1.4826 * np.median(np.abs(residual)))
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
    return solution[:2]


def normalized(values: np.ndarray) -> np.ndarray:
    low, high = np.quantile(values, [0.05, 0.95])
    return np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)


class PoseQualitySelector:
    def __init__(self, trajectory: Path, extrinsic: Path, receipt_path: Path):
        self.times, self.translations, rotations = read_trajectory(trajectory)
        self.rotation_slerp = Slerp(self.times, rotations)
        self.extrinsic_translation, self.extrinsic_rotation = read_extrinsic(extrinsic)
        self.receipt_path = receipt_path
        self.trajectory = trajectory
        self.extrinsic = extrinsic

    def camera_centers(self, timestamps_ns: np.ndarray) -> np.ndarray:
        if timestamps_ns.min() < self.times[0] or timestamps_ns.max() > self.times[-1]:
            raise RuntimeError("Candidate image timestamps exceed the metric trajectory support")
        columns = [
            np.interp(timestamps_ns, self.times, self.translations[:, axis])
            for axis in range(3)
        ]
        base_translation = np.column_stack(columns)
        base_rotation = self.rotation_slerp(timestamps_ns)
        return base_translation + base_rotation.apply(
            np.repeat(self.extrinsic_translation[None, :], len(timestamps_ns), axis=0)
        )

    def __call__(self, candidates: list[dict[str, Any]], target_count: int):
        if len(candidates) < target_count:
            raise ValueError("Candidate count is smaller than the target count")
        timestamps = np.asarray([item["camera_header_ns"] for item in candidates], dtype=np.float64)
        centers = self.camera_centers(timestamps)
        center_xy = robust_circle_center(centers[:, :2])
        offsets = centers[:, :2] - center_xy[None, :]
        azimuth = np.arctan2(offsets[:, 1], offsets[:, 0])
        radius = np.linalg.norm(offsets, axis=1)
        sharpness = np.asarray([item["sharpness"] for item in candidates], dtype=np.float64)
        clipping = np.asarray([item["clipped_fraction"] for item in candidates], dtype=np.float64)
        quality = normalized(np.log1p(sharpness)) * (1.0 - np.clip(clipping / 0.15, 0.0, 1.0))
        height = normalized(centers[:, 2])
        time_feature = normalized(timestamps)
        radius_feature = normalized(radius)
        features = np.column_stack(
            (
                np.cos(azimuth),
                np.sin(azimuth),
                1.6 * height,
                0.35 * radius_feature,
                0.30 * time_feature,
            )
        )
        eligible = (sharpness >= np.quantile(sharpness, 0.05)) & (
            clipping <= max(0.15, float(np.quantile(clipping, 0.98)))
        )
        selected: list[int] = []
        selected_set: set[int] = set()
        azimuth_degrees = np.degrees(azimuth) % 360.0
        candidate_azimuth_counts, _ = np.histogram(
            azimuth_degrees, bins=np.arange(0.0, 361.0, 30.0)
        )
        minimum_per_azimuth_bin = 10
        for azimuth_bin in range(12):
            lower = 30.0 * azimuth_bin
            upper = lower + 30.0
            pool = np.flatnonzero(
                eligible & (azimuth_degrees >= lower) & (azimuth_degrees < upper)
            )
            if len(pool) < minimum_per_azimuth_bin:
                raise RuntimeError(
                    f"Azimuth bin {azimuth_bin} has only {len(pool)} eligible candidates"
                )
            local_selected: list[int] = []
            while len(local_selected) < minimum_per_azimuth_bin:
                available = np.asarray(
                    [index for index in pool if int(index) not in selected_set], dtype=np.int64
                )
                if len(available) == 0:
                    raise RuntimeError(f"Cannot satisfy azimuth quota for bin {azimuth_bin}")
                score = quality[available] + 0.25 * height[available]
                if local_selected:
                    local_features = features[np.asarray(local_selected, dtype=np.int64)]
                    novelty = np.min(
                        np.sum(
                            (features[available, None, :] - local_features[None, :, :]) ** 2,
                            axis=2,
                        ),
                        axis=1,
                    )
                    score += 0.35 * novelty
                if selected:
                    nearest_time = np.min(
                        np.abs(timestamps[available, None] - timestamps[np.asarray(selected)][None, :]),
                        axis=1,
                    )
                    score[nearest_time < 100_000_000.0] = -np.inf
                if not np.isfinite(score).any():
                    score = quality[available] + 0.25 * height[available]
                choice = int(available[int(np.argmax(score))])
                selected.append(choice)
                selected_set.add(choice)
                local_selected.append(choice)
        seed_bins = min(24, target_count)
        boundaries = np.linspace(0, len(candidates), seed_bins + 1, dtype=np.int64)
        for bin_index in range(seed_bins):
            indices = np.arange(boundaries[bin_index], boundaries[bin_index + 1])
            valid = indices[eligible[indices]]
            if len(valid) == 0:
                valid = indices
            score = quality[valid] + 0.20 * height[valid]
            ordered = valid[np.argsort(score)[::-1]]
            for candidate_index in ordered:
                if int(candidate_index) not in selected_set:
                    selected.append(int(candidate_index))
                    selected_set.add(int(candidate_index))
                    break
        minimum_spacing_ns = 120_000_000.0
        while len(selected) < target_count:
            selected_features = features[np.asarray(selected, dtype=np.int64)]
            squared = np.sum((features[:, None, :] - selected_features[None, :, :]) ** 2, axis=2)
            novelty = np.min(squared, axis=1)
            score = novelty + 0.22 * quality + 0.28 * height
            score[~eligible] = -np.inf
            for index in selected_set:
                score[index] = -np.inf
            nearest_time = np.min(
                np.abs(timestamps[:, None] - timestamps[np.asarray(selected)][None, :]), axis=1
            )
            score[nearest_time < minimum_spacing_ns] = -np.inf
            if not np.isfinite(score).any():
                minimum_spacing_ns *= 0.75
                if minimum_spacing_ns < 20_000_000.0:
                    raise RuntimeError("Pose-diverse selector could not fill the requested count")
                continue
            choice = int(np.argmax(score))
            selected.append(choice)
            selected_set.add(choice)
        selected.sort(key=lambda index: timestamps[index])
        output = []
        for index in selected:
            row = dict(candidates[index])
            row["metric_camera_center_m"] = centers[index].tolist()
            row["orbit_azimuth_degrees"] = float(np.degrees(azimuth[index]) % 360.0)
            row["orbit_radius_m"] = float(radius[index])
            row["height_score"] = float(height[index])
            row["quality_score_normalized"] = float(quality[index])
            output.append(row)
        selected_centers = centers[np.asarray(selected)]
        azimuth_counts, _ = np.histogram(
            np.degrees(azimuth[np.asarray(selected)]) % 360.0, bins=np.arange(0.0, 361.0, 30.0)
        )
        receipt = {
            "schema_version": "e26_pose_quality_diverse_selection_v1",
            "status": "complete",
            "candidate_count": len(candidates),
            "selected_count": len(output),
            "method": "quality-gated greedy farthest-point selection over orbit azimuth, metric height, radius, and time",
            "seed_temporal_bins": seed_bins,
            "minimum_selected_per_azimuth_bin": minimum_per_azimuth_bin,
            "candidate_azimuth_30_degree_counts": candidate_azimuth_counts.tolist(),
            "sharpness_candidate_quantiles": np.quantile(sharpness, [0.0, 0.05, 0.5, 0.95, 1.0]).tolist(),
            "selected_sharpness_quantiles": np.quantile(sharpness[np.asarray(selected)], [0.0, 0.05, 0.5, 0.95, 1.0]).tolist(),
            "candidate_camera_z_quantiles_m": np.quantile(centers[:, 2], [0.0, 0.05, 0.5, 0.95, 1.0]).tolist(),
            "selected_camera_z_quantiles_m": np.quantile(selected_centers[:, 2], [0.0, 0.05, 0.5, 0.95, 1.0]).tolist(),
            "selected_azimuth_30_degree_counts": azimuth_counts.tolist(),
            "robust_orbit_center_xy_m": center_xy.tolist(),
            "trajectory": str(self.trajectory),
            "trajectory_sha256": sha256_file(self.trajectory),
            "extrinsic": str(self.extrinsic),
            "extrinsic_sha256": sha256_file(self.extrinsic),
            "language_model_used": False,
            "semantic_model_used": False,
            "manual_selection_used": False,
        }
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", required=True, type=Path)
    parser.add_argument("--mcap", required=True, type=Path)
    parser.add_argument("--sfm-tools", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--extrinsic", required=True, type=Path)
    parser.add_argument("--candidate-count", type=int, default=960)
    parser.add_argument("--image-count", type=int, default=180)
    parser.add_argument("--start-fraction", type=float, default=0.10)
    parser.add_argument("--end-fraction", type=float, default=0.68)
    parser.add_argument("--expected-mcap-sha256")
    parser.add_argument("--skip-full-mcap-hash", action="store_true")
    args = parser.parse_args()

    frontend = load_frontend(args.frontend.resolve(strict=True))
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selector = PoseQualitySelector(
        args.trajectory.resolve(strict=True),
        args.extrinsic.resolve(strict=True),
        output / "pose_quality_selection_receipt.json",
    )
    frontend.select_temporally_complete = selector
    manifest, calibration = frontend.extract_images(
        args.mcap.resolve(strict=True),
        args.sfm_tools.resolve(strict=True),
        output,
        args.candidate_count,
        args.image_count,
        args.start_fraction,
        args.end_fraction,
    )
    manifest_path = output / "image_manifest.json"
    calibration_path = output / "camera_calibration.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    frontend.write_contact_sheet(output / "images", manifest, output / "contact_sheet.png")
    receipt = {
        "schema_version": "e26_pose_quality_visual_workspace_v1",
        "status": "extracted",
        "mcap": {
            "path": str(args.mcap.resolve(strict=True)),
            "bytes": args.mcap.stat().st_size,
            "registered_sha256": args.expected_mcap_sha256,
            "full_hash_recomputed": not args.skip_full_mcap_hash,
        },
        "selection": {
            "candidate_count": args.candidate_count,
            "selected_image_count": len(manifest),
            "start_fraction": args.start_fraction,
            "end_fraction": args.end_fraction,
            "receipt": str((output / "pose_quality_selection_receipt.json").resolve()),
        },
        "color_contract": calibration["color_contract"],
        "image_manifest_sha256": sha256_file(manifest_path),
        "camera_calibration_sha256": sha256_file(calibration_path),
        "contact_sheet_sha256": sha256_file(output / "contact_sheet.png"),
        "language_model_used": False,
        "semantic_model_used": False,
        "manual_selection_used": False,
    }
    (output / "workspace_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
