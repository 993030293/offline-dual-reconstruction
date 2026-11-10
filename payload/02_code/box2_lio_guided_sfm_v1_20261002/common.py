"""Shared contracts for the isolated Box2 pose-refinement experiment."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pycolmap
from scipy.spatial.transform import Rotation


PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
BASE = PROJECT / "04_outputs/e26_box2_fresh_intervention_audit_20260928/box2_a1_raw/S00_S08"
VISUAL = BASE / "01_visual_workspace"
TOOLS = PACKAGE / "code/reference_sources/e26_c2_support/frontend/tools"
MCAP = Path("E:/lidar_reconstruction_video_data/raw_recordings/e26_orbit_capture_20260820_box_pair_v1/raw_mcap/20260820B_Box2/recon_collection_20260820_170743_689578_0.mcap")
COLMAP = PROJECT / "05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
PLAN = Path("C:/Users/99303/project-kb/vault/plans/project/lidar-camera-fusion-mask-robust-downsampling/box2_lio_guided_sfm_quality/r1_20261002T005049_9bdfd943/plan.md")


def configure_stdout() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name"):
        return value.name
    return value


def write_json(path: Path, value: Any, *, update: bool = False) -> None:
    if path.exists() and not update:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if update:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)


def digest_data(path: Path) -> str:
    """Hash data identity; never use this function to assess document content."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mcap_tools() -> tuple:
    sys.path.insert(0, str(TOOLS))
    from mcap_audit import McapIndex, Cdr
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
    return McapIndex, Cdr, decode_bayer, demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION


def image_quality(image: np.ndarray) -> dict:
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    return {"sharpness": sharpness, "clipped_fraction": clipped,
            "mean_brightness": float(gray.mean())}


class PoseCurve:
    """Interpolate T_world_body and compose the optical-camera lever arm."""

    def __init__(self, trajectory: Path, extrinsic: Path):
        data = np.loadtxt(trajectory)
        self.times = np.rint(data[:, 0] * 1e9).astype(np.int64)
        self.positions = data[:, 1:4]
        quaternions = data[:, 4:8]
        if len(data) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Trajectory timestamps are not strictly increasing")
        if not np.isfinite(data).all() or not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-6):
            raise ValueError("Invalid trajectory values or quaternion norm")
        self.rotations = Rotation.from_quat(quaternions)
        self.quaternions = self.rotations.as_quat()
        self.origin = int(self.times[0])
        geometry = PACKAGE / "code/reference_sources/e26_c2_support/geometry/src"
        sys.path.insert(0, str(geometry))
        from lidar_camera_recon.e26_paperbox_precision import slerp_xyzw
        self.project_slerp = slerp_xyzw
        item = read_json(extrinsic)
        if item["source_frame"] != "camera_optical" or item["target_frame"] != "base_link":
            raise ValueError("Unexpected extrinsic direction")
        pose = np.asarray(item["pose_xyz_xyzw"], dtype=float)
        if not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 1e-6:
            raise ValueError("Invalid camera extrinsic")
        self.ext_t = pose[:3]
        self.ext_r = Rotation.from_quat(pose[3:])

    def at(self, timestamp_ns: int, maximum_span_ns: int = 120_000_000) -> dict:
        stamp = int(timestamp_ns)
        if stamp < self.times[0] or stamp > self.times[-1]:
            raise ValueError("Extrapolation is forbidden")
        right = int(np.searchsorted(self.times, stamp, side="left"))
        if self.times[right] == stamp:
            left = right
        else:
            left = right - 1
        span = int(self.times[right] - self.times[left])
        if span > maximum_span_ns:
            raise ValueError("Interpolation bracket exceeds the frozen policy")
        alpha = 0.0 if span == 0 else (stamp - int(self.times[left])) / span
        body_t = (1 - alpha) * self.positions[left] + alpha * self.positions[right]
        body_r = Rotation.from_quat(self.project_slerp(
            self.quaternions[left], self.quaternions[right], alpha))
        camera_r = body_r * self.ext_r
        center = body_t + body_r.apply(self.ext_t)
        return {"camera_header_ns": stamp, "adjusted_timestamp_ns": stamp,
                "time_offset_ns": 0, "time_offset_source": "existing_sensor_header_contract_not_independently_calibrated",
                "left_pose_index": left, "right_pose_index": right,
                "left_pose_timestamp_ns": int(self.times[left]),
                "right_pose_timestamp_ns": int(self.times[right]),
                "bracket_span_ns": span, "alpha": float(alpha),
                "C_W_m": center.tolist(), "R_WC": camera_r.as_matrix().tolist(),
                "quaternion_WC_xyzw": camera_r.as_quat().tolist(),
                "extrapolated": False, "offline_future_bracketing": True}


def metric_initial_model() -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction(VISUAL / "sparse/0")
    sim = read_json(BASE / "05_sim3/sim3.json")
    reconstruction.transform(pycolmap.Sim3d(float(sim["scale"]),
        pycolmap.Rotation3d(np.asarray(sim["rotation_xyzw"])),
        np.asarray(sim["translation_m"])))
    return reconstruction


def validate_prior_rows(reconstruction: pycolmap.Reconstruction, rows: list[dict]) -> None:
    ids = [int(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prior image IDs")
    for row in rows:
        image_id = int(row["image_id"])
        if image_id not in reconstruction.images:
            raise ValueError("Prior refers to an unknown image ID")
        image = reconstruction.images[image_id]
        if row["image_name"] != image.name:
            raise ValueError("Prior image identity mismatch")
        center = np.asarray(row["C_W_m"], dtype=float)
        covariance = np.asarray(row["position_covariance_m2"], dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Nonfinite prior position")
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            raise ValueError("Invalid prior covariance")
        if not np.allclose(covariance, covariance.T, atol=1e-12) or np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("Prior covariance must be positive definite")
    active = [row for row in rows if row.get("prior_active", True)]
    if len(active) < 3:
        raise ValueError("At least three active position priors are required")
    centers = np.asarray([r["C_W_m"] for r in active])
    if np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)[1] < 1e-7:
        raise ValueError("Active position priors are collinear")


def make_priors(reconstruction: pycolmap.Reconstruction, rows: list[dict],
                sigma_m: float | None = None) -> list:
    validate_prior_rows(reconstruction, rows)
    priors = []
    for row in rows:
        if not row.get("prior_active", True):
            continue
        covariance = np.asarray(row["position_covariance_m2"])
        if sigma_m is not None:
            covariance = np.eye(3) * sigma_m**2
        image = reconstruction.images[int(row["image_id"])]
        priors.append(pycolmap.PosePrior(
            pose_prior_id=len(priors) + 1, corr_data_id=image.data_id,
            position=np.asarray(row["C_W_m"]), position_covariance=covariance,
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN))
    return priors


def ba_options(iterations: int) -> pycolmap.BundleAdjustmentOptions:
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    options.refine_rig_from_world = True
    options.refine_points3D = True
    options.print_summary = True
    options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    options.ceres.loss_function_scale = 1.0
    options.ceres.solver_options.max_num_iterations = iterations
    options.ceres.solver_options.max_solver_time_in_seconds = 1800
    options.ceres.solver_options.num_threads = 8
    options.ceres.solver_options.function_tolerance = 1e-9
    options.ceres.solver_options.parameter_tolerance = 1e-10
    return options


def ba_config(reconstruction: pycolmap.Reconstruction, fixed_poses: bool = False):
    config = pycolmap.BundleAdjustmentConfig()
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        config.add_image(image.image_id)
        config.set_constant_cam_intrinsics(image.camera_id)
        if fixed_poses:
            config.set_constant_rig_from_world_pose(image.frame_id)
    return config


def solver_receipt(summary: Any) -> dict:
    native = summary.ceres_summary
    return {"usable": bool(summary.is_solution_usable()), "termination_type": str(summary.termination_type),
            "brief_report": summary.brief_report(), "num_residual_blocks": int(native.num_residual_blocks),
            "num_residuals": int(native.num_residuals),
            "num_residual_blocks_reduced": int(native.num_residual_blocks_reduced),
            "num_residuals_reduced": int(native.num_residuals_reduced),
            "initial_cost": float(native.initial_cost), "final_cost": float(native.final_cost),
            "native_summary": summary.todict()}


class Heartbeat:
    """Write process-local progress without touching any other job."""

    def __init__(self, path: Path):
        self.path, self.start = path, time.monotonic()
        self.phase = "starting"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            process = psutil.Process()
            write_json(self.path, {"pid": os.getpid(), "phase": self.phase,
                "elapsed_seconds": time.monotonic() - self.start,
                "rss_bytes": process.memory_info().rss, "timestamp_unix": time.time()}, update=True)
            self.stop.wait(20)

    def close(self):
        self.stop.set()
        self.thread.join()
