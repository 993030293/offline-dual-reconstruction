"""Check that an ROI fringe is removed only when sparse target support survives."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "run_reliable_lidar_tsdf.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_reliable_lidar_tsdf", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def roi_receipt() -> dict:
    return {
        "roi_derivation": {
            "metric_sparse_vehicle_support": {
                "quantile_minimum_xyz_m": [0.0, 0.0, 0.0],
                "quantile_maximum_xyz_m": [1.0, 1.0, 1.0],
            }
        },
        "roi_world_lio_m": {
            "minimum": [-0.15, -0.15, -0.15],
            "maximum": [1.15, 1.15, 1.15],
        },
    }


def surface_with_fringe() -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    target = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    fringe = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.20, depth=0.20)
    fringe.translate((-0.08, 0.2, 0.2))
    return target + fringe, np.asarray(target.vertices).copy()


class AdaptiveRoiTrimTest(unittest.TestCase):
    def test_unsupported_fringe_is_trimmed_at_smallest_padding(self) -> None:
        surface, support = surface_with_fringe()
        trimmed, audit = MODULE.trim_surface_roi_by_sparse_support(
            surface, support, roi_receipt()
        )
        self.assertEqual(audit["selected_padding_m"], 0.03)
        self.assertLess(len(trimmed.triangles), len(surface.triangles))
        self.assertEqual(audit["candidates"][0]["absolute_support_drop"], 0.0)

    def test_supported_fringe_keeps_enough_padding(self) -> None:
        surface, support = surface_with_fringe()
        support = np.vstack((support, [-0.11, 0.2, 0.2]))
        trimmed, audit = MODULE.trim_surface_roi_by_sparse_support(
            surface, support, roi_receipt()
        )
        self.assertEqual(audit["selected_padding_m"], 0.10)
        self.assertEqual(len(trimmed.triangles), len(surface.triangles))
        self.assertEqual(
            [row["status"] for row in audit["candidates"]],
            ["support_loss", "support_loss", "accepted"],
        )


if __name__ == "__main__":
    unittest.main()
