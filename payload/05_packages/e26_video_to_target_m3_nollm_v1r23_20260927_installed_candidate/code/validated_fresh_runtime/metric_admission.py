"""Pure machine gates for independent sparse-camera geometry support."""

from __future__ import annotations

import numpy as np


def camera_geometry_metrics(reconstruction, image, policy: dict) -> dict:
    camera = reconstruction.cameras[image.camera_id]
    observations = np.asarray(
        [point.xy for point in image.points2D if point.has_point3D()], dtype=np.float64
    )
    if len(observations) == 0:
        bbox_area_fraction = 0.0
        occupied_cells = 0
    else:
        clipped = np.column_stack(
            (
                np.clip(observations[:, 0], 0.0, float(camera.width)),
                np.clip(observations[:, 1], 0.0, float(camera.height)),
            )
        )
        span = clipped.max(axis=0) - clipped.min(axis=0)
        bbox_area_fraction = float(
            (span[0] / float(camera.width)) * (span[1] / float(camera.height))
        )
        rows = int(policy["grid_rows"])
        columns = int(policy["grid_columns"])
        cells = {
            (
                min(columns - 1, int(columns * x / float(camera.width))),
                min(rows - 1, int(rows * y / float(camera.height))),
            )
            for x, y in clipped
            if x < camera.width and y < camera.height
        }
        occupied_cells = len(cells)
    passed = (
        len(observations) >= int(policy["minimum_triangulated_points"])
        and occupied_cells >= int(policy["minimum_occupied_grid_cells"])
        and bbox_area_fraction >= float(policy["minimum_bbox_area_fraction"])
    )
    return {
        "image_name": image.name,
        "triangulated_points": len(observations),
        "occupied_grid_cells": occupied_cells,
        "bbox_area_fraction": bbox_area_fraction,
        "passed": passed,
    }
