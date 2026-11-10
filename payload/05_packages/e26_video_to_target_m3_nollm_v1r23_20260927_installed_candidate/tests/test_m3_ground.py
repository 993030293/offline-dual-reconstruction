"""Regression checks for the frozen ground graph's fail-closed boundaries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ground" / "code"))
import m3_ground


class M3GroundTests(unittest.TestCase):
    def test_flat_exterior_ground_with_raised_center(self) -> None:
        floor = np.zeros((12, 12), dtype=np.float64)
        floor[4:8, 4:8] = 0.10
        connected, _ = m3_ground.connected_ground_cells(floor)
        mrf, _ = m3_ground.mrf_ground_cells(floor)
        self.assertTrue(np.all(connected[:2]))
        self.assertTrue(np.all(mrf[:2]))
        self.assertFalse(np.any(connected[4:8, 4:8]))
        self.assertFalse(np.any(mrf[4:8, 4:8]))

    def test_no_exterior_ground_abstains(self) -> None:
        floor = np.full((8, 8), np.inf, dtype=np.float64)
        floor[2:6, 2:6] = 0.0
        with self.assertRaisesRegex(RuntimeError, "exterior"):
            m3_ground.connected_ground_cells(floor)
        with self.assertRaisesRegex(RuntimeError, "exterior"):
            m3_ground.mrf_ground_cells(floor)


if __name__ == "__main__":
    unittest.main()
