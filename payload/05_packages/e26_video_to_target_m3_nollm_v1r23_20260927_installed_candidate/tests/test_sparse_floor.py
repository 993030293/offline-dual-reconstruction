"""Regression checks for floor selection under stacked target surfaces."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from estimate_sparse_floor import estimate_sparse_floor


class SparseFloorTests(unittest.TestCase):
    def test_spatial_floor_beats_dense_raised_box_top(self) -> None:
        grid = np.arange(-1.0, 1.01, 0.1)
        x, y = np.meshgrid(grid, grid, indexing="ij")
        floor = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.2)))
        box_x, box_y = np.meshgrid(np.arange(-0.35, 0.36, 0.025), np.arange(-0.35, 0.36, 0.025))
        raised = np.column_stack((box_x.ravel(), box_y.ravel(), np.full(box_x.size, -0.55)))
        points = np.vstack((floor, raised))
        result = estimate_sparse_floor(points, np.zeros(2), 0.15, 3.0)
        self.assertAlmostEqual(result["ground_z_at_focus_m"], -1.2, delta=0.01)
        self.assertGreaterEqual(result["inlier_cell_count"], 60)

    def test_no_spatial_floor_fails_closed(self) -> None:
        x, y = np.meshgrid(np.arange(-0.3, 0.31, 0.01), np.arange(-0.3, 0.31, 0.01))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -0.5)))
        with self.assertRaises(RuntimeError):
            estimate_sparse_floor(points, np.zeros(2), 0.15, 3.0)


if __name__ == "__main__":
    unittest.main()
