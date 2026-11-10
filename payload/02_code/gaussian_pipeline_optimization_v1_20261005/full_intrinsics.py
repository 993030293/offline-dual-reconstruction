"""Preserve COLMAP corner-based intrinsics in the existing INRIA renderer."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np


@dataclass(frozen=True)
class Pinhole:
    """Use corner-based image coordinates with the first centre at (0.5, 0.5)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
        values = [self.fx, self.fy, self.cx, self.cy]
        if not np.isfinite(values).all() or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Invalid calibrated pinhole intrinsics")

    @classmethod
    def from_colmap(cls, camera):
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        else:
            raise ValueError("Undistort to PINHOLE or SIMPLE_PINHOLE before training")
        return cls(int(camera.width), int(camera.height), *map(float, (fx, fy, cx, cy)))

    def resized(self, width: int, height: int):
        # Corner-based coordinates scale directly; subtracting 0.5 here is wrong.
        sx, sy = width / self.width, height / self.height
        return Pinhole(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)

    def matrix(self):
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def projection(self, znear=0.01, zfar=100.0):
        if not 0 < znear < zfar:
            raise ValueError("Invalid depth clipping interval")
        result = np.zeros((4, 4), dtype=np.float64)
        result[0, 0] = 2.0 * self.fx / self.width
        result[1, 1] = 2.0 * self.fy / self.height
        result[0, 2] = 2.0 * self.cx / self.width - 1.0
        result[1, 2] = 2.0 * self.cy / self.height - 1.0
        result[2, 2] = zfar / (zfar - znear)
        result[2, 3] = -zfar * znear / (zfar - znear)
        result[3, 2] = 1.0
        return result


def apply_to_camera(camera, intrinsics: Pinhole):
    """Set full projection while preserving the original metric view transform."""
    import torch

    if (camera.image_width, camera.image_height) != (intrinsics.width, intrinsics.height):
        raise ValueError("Image and intrinsics resolutions differ")
    camera.FoVx = 2.0 * math.atan(intrinsics.width / (2.0 * intrinsics.fx))
    camera.FoVy = 2.0 * math.atan(intrinsics.height / (2.0 * intrinsics.fy))
    view = camera.world_view_transform
    camera.projection_matrix = torch.as_tensor(
        intrinsics.projection(camera.znear, camera.zfar), device=view.device, dtype=view.dtype
    ).T.contiguous()
    camera.full_proj_transform = view @ camera.projection_matrix
    camera.calibrated_intrinsics = asdict(intrinsics)
    camera.pixel_convention = "colmap_corner_first_center_0.5"
    return camera


@contextmanager
def camera_loader_adapter(dataset: Path, enabled=True):
    """Adapt the loaded camera objects without changing the frozen source tree."""
    import utils.camera_utils as camera_utils
    from scene.colmap_loader import read_intrinsics_binary

    original = camera_utils.loadCam
    cameras = read_intrinsics_binary(Path(dataset) / "sparse/0/cameras.bin")
    intrinsics = {camera_id: Pinhole.from_colmap(camera) for camera_id, camera in cameras.items()}
    loaded = []

    def calibrated_load(args, camera_id, info, resolution_scale, is_synthetic, is_test):
        if is_synthetic:
            raise ValueError("This adapter requires calibrated COLMAP cameras")
        intrinsic = intrinsics[info.uid]
        if (info.width, info.height) != (intrinsic.width, intrinsic.height):
            raise ValueError("COLMAP camera metadata has inconsistent dimensions")
        camera = original(args, camera_id, info, resolution_scale, is_synthetic, is_test)
        scaled = intrinsic.resized(camera.image_width, camera.image_height)
        if enabled:
            apply_to_camera(camera, scaled)
        loaded.append({"name": info.image_name, "camera_id": info.uid,
                       "intrinsics": asdict(scaled), "full_intrinsics_applied": enabled})
        return camera

    camera_utils.loadCam = calibrated_load
    try:
        yield loaded
    finally:
        camera_utils.loadCam = original
