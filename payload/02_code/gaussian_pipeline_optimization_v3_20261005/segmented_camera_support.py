"""Reduce complete native camera support by image segments without changing gates."""

import time
from pathlib import Path

import numpy as np
import pycolmap

from native_observation_index import NativeObservationIndex, admission, snapshot, image_arrays


class SupportObservationIndex(NativeObservationIndex):
    """Bind serialized registered rows and omitted image fields to one live state."""

    @staticmethod
    def omitted_images(model):
        result = {}
        for image_id, image in model.images.items():
            if image.has_pose:
                continue
            points = list(image.points2D)
            pixels = np.asarray([point.xy for point in points], dtype=np.float64).reshape(-1, 2)
            ids = np.asarray([int(point.point3D_id) for point in points], dtype=np.uint64)
            result[image_id] = (image.name, image.camera_id, image.frame_id, image.num_points2D(),
                                pixels.tobytes(), ids.tobytes())
        return result

    def __init__(self, model, output):
        compatible, reason = admission(model)
        if not compatible:
            raise ValueError("An unadmitted map cannot allocate a support index: " + reason)
        started = time.perf_counter()
        self.model, self.output = model, Path(output)
        self.omitted_before = self.omitted_images(model)
        self.omitted_verified = False
        self.before = snapshot(model, self.output/"before")
        self.rows = {row["id"]: row for row in image_arrays(self.output/"before/images.bin")}
        registered = {image_id for image_id, image in model.images.items() if image.has_pose}
        if set(self.rows) != registered:
            raise ValueError("The complete registered support snapshot identities changed")
        self.cache, self.hits, self.misses = {}, 0, 0
        self.bytes = sum(row["data"].nbytes for row in self.rows.values())
        self.bytes += sum(len(row[-1])+len(row[-2]) for row in self.omitted_before.values())
        if self.bytes > 128*1024**2:
            raise MemoryError("The complete registered and omitted support state exceeds its budget")
        self.build_seconds = time.perf_counter()-started
        self.verify_seconds, self.verified = 0., False

    def verify_unchanged(self):
        started = time.perf_counter()
        super().verify_unchanged()
        if self.omitted_before != self.omitted_images(self.model):
            self.verified = False
            raise ValueError("An omitted unregistered image changed during owned camera support")
        self.omitted_verified = True
        self.verify_seconds = time.perf_counter()-started

    def receipt(self):
        result = super().receipt()
        result.update(unregistered_images=len(self.omitted_before),
                      omitted_image_metadata_pixels_and_bindings_exact=self.omitted_verified,
                      expected_native_snapshot_registered_only=True)
        return result


def policy_supported(policy):
    if type(policy) is not dict:
        return False
    integer_keys = ("grid_rows", "grid_columns", "minimum_triangulated_points", "minimum_occupied_grid_cells")
    if any(type(policy.get(key)) is not int for key in integer_keys):
        return False
    rows, columns = policy["grid_rows"], policy["grid_columns"]
    fraction = policy.get("minimum_bbox_area_fraction")
    return (0 < rows <= 65536 and 0 < columns <= 65536 and rows * columns <= 2**32
            and type(fraction) in (int, float) and np.isfinite(fraction))


def segmented_rows(names, dimensions, observed, policy):
    """Keep scalar operation order while grouping independent finite observations."""
    if not policy_supported(policy) or len(names) != len(dimensions) or len(names) != len(observed):
        raise ValueError("Segmented support requires its admitted policy and complete camera lists")
    if not names:
        return []
    sizes = np.asarray(dimensions, dtype=np.float64)
    if sizes.shape != (len(names), 2) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("Segmented support requires finite positive native dimensions")
    arrays = [np.asarray(points) for points in observed]
    if any(points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 2
           or not np.isfinite(points).all() for points in arrays):
        raise ValueError("Segmented support requires complete finite float64 pixel arrays")
    counts = np.asarray([len(points) for points in arrays], dtype=np.int64)
    total = int(counts.sum())
    if total > 5_000_000 or len(names) > 100_000:
        raise MemoryError("The complete segmented observation budget was exceeded")
    rows, columns = policy["grid_rows"], policy["grid_columns"]
    cells_per_image = rows * columns
    if len(names) * cells_per_image > np.iinfo(np.int64).max:
        raise MemoryError("The composite image and grid cell identity exceeds int64")
    occupied = np.zeros(len(names), dtype=np.int64)
    bbox = np.zeros(len(names), dtype=np.float64)
    if total:
        pixels = np.concatenate(arrays, axis=0)
        labels = np.repeat(np.arange(len(names), dtype=np.int64), counts)
        width, height = sizes[labels, 0], sizes[labels, 1]
        clipped = np.column_stack((np.clip(pixels[:, 0], 0.0, width),
                                   np.clip(pixels[:, 1], 0.0, height)))
        nonempty = np.flatnonzero(counts)
        starts = np.cumsum(counts) - counts
        minimum = np.minimum.reduceat(clipped, starts[nonempty], axis=0)
        maximum = np.maximum.reduceat(clipped, starts[nonempty], axis=0)
        span = maximum - minimum
        bbox[nonempty] = (span[:, 0] / sizes[nonempty, 0]) * (span[:, 1] / sizes[nonempty, 1])
        mask = (clipped[:, 0] < width) & (clipped[:, 1] < height)
        selected = clipped[mask]
        x = np.minimum(columns - 1, (columns * selected[:, 0] / width[mask]).astype(np.int64))
        y = np.minimum(rows - 1, (rows * selected[:, 1] / height[mask]).astype(np.int64))
        keys = labels[mask] * cells_per_image + y * columns + x
        unique_image_ids = np.unique(keys) // cells_per_image
        occupied = np.bincount(unique_image_ids, minlength=len(names))
    result = []
    for name, count, cells, fraction in zip(names, counts, occupied, bbox):
        count, cells, fraction = int(count), int(cells), float(fraction)
        passed = (count >= int(policy["minimum_triangulated_points"])
                  and cells >= int(policy["minimum_occupied_grid_cells"])
                  and fraction >= float(policy["minimum_bbox_area_fraction"]))
        result.append({"image_name": name, "triangulated_points": count,
                       "occupied_grid_cells": cells, "bbox_area_fraction": fraction, "passed": passed})
    return result


def native_support_many(model, images, policy, output, original):
    """Build and verify one state; never reuse support across map mutations."""
    started = time.perf_counter()
    images = tuple(images)
    trace = {"images": len(images), "route": "native_before_execution", "index": None,
             "closed": False, "complete_seconds": None}
    index = None
    compatible, reason = admission(model) if type(model) is pycolmap.Reconstruction else (False, "native_type")
    supported = (compatible and policy_supported(policy)
                 and all(type(image) is pycolmap.Image and image.has_pose and image.image_id in model.images
                         and image is model.images[image.image_id] for image in images))
    supported = supported and sum(image.num_points2D() for image in images) <= 5_000_000
    dimensions = [(model.cameras[image.camera_id].width, model.cameras[image.camera_id].height)
                  for image in images] if supported else []
    supported = supported and all(width > 0 and height > 0 for width, height in dimensions)
    try:
        if not supported:
            trace["reason"] = reason if not compatible else "policy_image_ownership_or_dimensions"
            return [original(model, image, policy) for image in images], trace
        index = SupportObservationIndex(model, output)
        if any(not np.isfinite(index.rows[image.image_id]["data"]["xy"]).all()
               or np.any(index.rows[image.image_id]["data"]["point_id"] < -1) for image in images):
            trace["reason"] = "nonfinite_pixels_or_unsigned_ids_original_function_retained"
            return [original(model, image, policy) for image in images], trace
        observed = []
        for image in images:
            pixels, _, valid = index.get(model, image.image_id)
            observed.append(pixels[valid])
        result = segmented_rows([image.name for image in images], dimensions, observed, policy)
        trace.update(route="whole_map_segmented_support", reason="complete_owned_native_state_and_finite_pixels")
        return result, trace
    finally:
        try:
            if index is not None:
                index.verify_unchanged()
                trace["index"] = index.receipt()
        finally:
            trace.update(closed=True, complete_seconds=time.perf_counter() - started)
