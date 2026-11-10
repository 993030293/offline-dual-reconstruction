"""Apply a bounded object-volume hint without removing background supervision."""

import time

import numpy as np

from full_intrinsics import Pinhole


def volume_ray_mask(intrinsic, camera_to_world, minimum, maximum, near_depth=.2):
    """Intersect COLMAP pixel-center rays with a metric axis-aligned volume."""
    transform = np.asarray(camera_to_world, dtype=np.float64)
    lower, upper = np.asarray(minimum, dtype=np.float64), np.asarray(maximum, dtype=np.float64)
    if (transform.shape != (4, 4) or lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(transform).all() or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower >= upper) or not np.isfinite(near_depth) or near_depth <= 0):
        raise ValueError("The camera, metric volume, or clipping depth is invalid")
    rotation, origin = transform[:3, :3], transform[:3, 3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=3e-5, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=3e-5)
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError("A metric rigid camera pose is required")
    x = (np.arange(intrinsic.width, dtype=np.float64)[None, :]+.5-intrinsic.cx)/intrinsic.fx
    y = (np.arange(intrinsic.height, dtype=np.float64)[:, None]+.5-intrinsic.cy)/intrinsic.fy
    near = np.full((intrinsic.height, intrinsic.width), near_depth, dtype=np.float64)
    far = np.full_like(near, np.inf)
    valid = np.ones_like(near, dtype=bool)
    for axis in range(3):
        direction = rotation[axis, 0]*x+rotation[axis, 1]*y+rotation[axis, 2]
        parallel = np.abs(direction) <= 1e-12
        if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
            valid &= ~parallel
        lo = np.divide(lower[axis]-origin[axis], direction, out=np.full_like(near, -np.inf), where=~parallel)
        hi = np.divide(upper[axis]-origin[axis], direction, out=np.full_like(near, np.inf), where=~parallel)
        near = np.maximum(near, np.minimum(lo, hi))
        far = np.minimum(far, np.maximum(lo, hi))
    return np.ascontiguousarray(valid & (far >= near), dtype=np.uint8)


def weighted_l1(prediction, reference, mask):
    """Use weights two inside the volume hint and one everywhere else."""
    import torch

    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[0] != 3:
        raise ValueError("This registered objective requires one RGB training image")
    if mask.shape != prediction.shape[1:]:
        raise ValueError("The target volume hint and actual render sizes disagree")
    weights = 1 + mask.to(device=prediction.device, dtype=prediction.dtype)
    return ((prediction-reference).abs()*weights[None]).sum()/(3*weights.sum())


class TargetVolumeLoss:
    """Bind conservative ray-volume hints to actual immutable training cameras."""

    def __init__(self, cameras, minimum, maximum):
        import torch

        started = time.monotonic()
        self.records = {}
        self.calls = 0
        self.current = None
        for camera in cameras:
            intrinsic = Pinhole(**camera.calibrated_intrinsics)
            camera_to_world = np.linalg.inv(camera.world_view_transform.detach().cpu().numpy().T.astype(np.float64))
            mask = volume_ray_mask(intrinsic, camera_to_world, minimum, maximum)
            if camera.original_image.shape != (3, intrinsic.height, intrinsic.width):
                raise ValueError("The image and full calibration differ")
            if id(camera) in self.records:
                raise ValueError("Repeated camera object in the target-volume input")
            self.records[id(camera)] = {"name": camera.image_name, "mask": torch.from_numpy(mask),
                "pixels": int(mask.sum()), "total_pixels": int(mask.size), "shape": tuple(camera.original_image.shape)}
        if not self.records:
            raise ValueError("The target-volume input has no cameras")
        self.preparation_seconds = time.monotonic()-started

    def select(self, camera):
        if self.current is not None:
            raise ValueError("A selected training image did not consume its weighted loss")
        if id(camera) not in self.records:
            raise ValueError("Unregistered training camera for target weighting")
        self.current = self.records[id(camera)]

    def loss(self, prediction, reference):
        if self.current is None or tuple(prediction.shape) != self.current["shape"]:
            raise ValueError("The weighted objective lost its render/camera correspondence")
        value = weighted_l1(prediction, reference, self.current["mask"])
        self.calls += 1
        self.current = None
        return value

    def finish(self, iterations):
        if self.current is not None or self.calls != iterations:
            raise ValueError("The weighted objective did not cover the exact training denominator")
        return {"calls": self.calls, "cameras": len(self.records), "foreground_weight": 2, "background_weight": 1,
                "normalization": "sum_RGB_abs_error_times_weight_divided_by_3_sum_spatial_weight",
                "near_camera_depth_m": .2, "occlusion_resolved": False, "instance_segmentation_claim": False,
                "preparation_seconds": self.preparation_seconds,
                "rows": [{key: row[key] for key in ("name", "pixels", "total_pixels")} for row in self.records.values()]}
